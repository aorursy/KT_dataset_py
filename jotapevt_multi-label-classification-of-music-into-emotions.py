%load_ext tensorboard
# Data Preprocessing Packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
from scipy.io.arff import loadarff

# Data Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns

# ML Packages
from sklearn import model_selection, svm
from sklearn.svm import SVR
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import time
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPool1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import class_weight
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import multilabel_confusion_matrix,ConfusionMatrixDisplay, f1_score
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from sklearn import metrics

raw_data = loadarff('../input/multilabel-classification-emotions/emotions.arff')
df = pd.DataFrame(raw_data[0]).astype(float)
#df.drop(columns = ['Unnamed: 0'],inplace = True)
df.head()
df.info() #Acquiring basic information of each columns
df.describe()
labels = ['amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still', 'sad-lonely', 'angry-aggresive']

feats_df = df.drop(columns = labels)
labels_df = df[labels]
label_d = {'amazed-suprised' : 0, 'happy-pleased': 0, 'relaxing-calm' : 0, 'quiet-still' : 0, 'sad-lonely': 0, 'angry-aggresive' : 0}
for col in labels_df.columns:
    label_d[col] += len(labels_df[labels_df[col] == 1])

plt.figure(figsize = (16,8))
plt.bar(range(len(label_d)), list(label_d.values()), align='center')
plt.xticks(range(len(label_d)), list(label_d.keys()))
plt.show()
def transform_multiclass(orig_df):
    df = orig_df.copy()
    classe = []
    for i in range(len(df)):
        classe.append('')
    
    for i in range(len(df)):
        if df['amazed-suprised'][i] == 1:
            classe[i] = classe[i] + 'surprised-'
        if df['happy-pleased'][i] == 1:
            classe[i] = classe[i] + 'happy-'
        if df['relaxing-calm'][i] == 1:
            classe[i] = classe[i] + 'relaxing-'
        if df['quiet-still'][i] == 1:
            classe[i] = classe[i] + 'still-'
        if df['sad-lonely'][i] == 1:
            classe[i] = classe[i] + 'lonely-'
        if df['angry-aggresive'][i] == 1:
            classe[i] = classe[i] + 'angry-'
            
    df['Class'] = classe
    df.drop(['angry-aggresive','amazed-suprised','happy-pleased','relaxing-calm','quiet-still','sad-lonely'],axis=1,inplace = True)
    return df
class_label_df = transform_multiclass(labels_df)
class_label_df['Class'].unique()
plt.figure(figsize = (16,8))
sns.countplot(y = class_label_df.Class)
plt.show()
class_label_df.Class.value_counts()
df.corr().style.background_gradient(cmap='coolwarm')
df.duplicated().unique()
df.isnull().sum().unique()
# Correlational Matrix
corr_matrix = feats_df.corr().abs()

# Selects Upper section of the matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Finds column with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

#Remove highly correlated features
df.drop(df[to_drop], axis=1,inplace = True)
df['Class'] = class_label_df['Class']
df.head()
class_df = df.drop(labels,1)
print(class_df.shape)
class_df.head()
classes = class_label_df.Class.value_counts().index
class_series =class_label_df['Class'].value_counts()
outliers = []
for label in classes:
    if class_series[label] < 5:
        outliers.append(label)
outliers
for outlier in outliers:
    class_df = class_df[class_df['Class'] != outlier]

print(class_df.shape)
X = class_df.drop('Class',1)
y = class_df['Class']

print(X.shape, y.shape)
ros = RandomOverSampler()
X_over, y_over = ros.fit_resample(X, y)
print(X_over.shape)
print(y_over.shape)
y_over.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X_over,y_over,test_size = 0.2, random_state = 42)

train_df = X_train.join(y_train)
test_df = X_test.join(y_test)
test_df.shape
test_df.duplicated().sum()
test_df.drop_duplicates(inplace = True)
test_df.shape
train_df.shape
le = LabelEncoder()

X_train = train_df.drop('Class',1).values
X_train = preprocessing.scale(X_train)
y_train = le.fit_transform(train_df['Class'])

X_test = test_df.drop('Class',1).values
X_test = preprocessing.scale(X_test)
y_test = le.fit_transform(test_df['Class'])

X_train
classifier = RandomForestClassifier(n_estimators = 100,n_jobs = -1, verbose = 0, random_state = 30)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
f1 = round(f1_score(y_test,y_pred, average = 'micro'),2)
accuracy = round(accuracy_score(y_test,y_pred),2)

print(f'Accuracy: {accuracy*100}%\n')
print(f'F1 Score: {f1*100}%\n')
print(f'Confusion Matrix:\n{cm}')
#xgb_params = {
#    'objective': 'multi:softmax'
#}

classifier = OneVsRestClassifier(xgb.XGBClassifier(n_jobs = -1))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
f1 = round(f1_score(y_test,y_pred, average = 'micro'),3)
accuracy = round(accuracy_score(y_test,y_pred),3)

print(f'Accuracy: {accuracy*100}%\n')
print(f'F1 Score: {f1*100}%\n')
print(f'Confusion Matrix:\n{cm}')
BATCH_SIZE = 64
EPOCHS = 100
dense_layers = [1]
conv_layers = [2]
layer_sizes = [128]

X_train_n = X_train
y_train_n = y_train
X_test_n = X_test
y_test_n = y_test


X_train_n = preprocessing.scale(X_train_n)
X_test_n = preprocessing.scale(X_test_n)

X_train_n = np.reshape(X_train_n, (X_train_n.shape[0], X_train_n.shape[1] , 1))
X_test_n = np.reshape(X_test_n, (X_test_n.shape[0], X_test_n.shape[1], 1))
sw = class_weight.compute_sample_weight('balanced', y_train_n)

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f'emotions_cnn-dense-{dense_layer}-layer{layer_size}-conv{conv_layer}-time-{int(time.time())}'
            tb_callback = TensorBoard(log_dir = f'logs/{NAME}')
            
            model2 = Sequential()
            
            if conv_layer > 0:
                model2.add(Conv1D(layer_size, 3,padding='valid', input_shape=(53,1),activation='tanh'))
                model2.add(BatchNormalization())
                model2.add(MaxPooling1D(pool_size=(4)))
                model2.add(Dropout(0.2)) 
                
                for l in range(conv_layer - 1):
                    model2.add(Conv1D(layer_size, 3,padding='valid',activation='tanh'))
                    model2.add(BatchNormalization())
                    model2.add(MaxPooling1D(pool_size=(2)))
                    model2.add(Dropout(0.2)) 
            
                                   
            else:
                model2.add(Flatten())
                model2.add(Dense(layer_size, input_shape = (53,) ,activation='tanh'))
            
            model2.add(Dropout(0.2))  
            model2.add(Flatten())
            for l in range(dense_layer):
                model2.add(Dense(layer_size, activation='relu'))
                model2.add(BatchNormalization())
                model2.add(Dropout(0.1))
                
            model2.add(Dropout(0.3))
            model2.add(Dense(20, activation="softmax"))

            model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model2.fit(X_train_n, y_train_n, 
                                 batch_size=BATCH_SIZE, 
                                 epochs=EPOCHS,
                                 verbose= 0,
                                 validation_data=(X_test_n, y_test_n),
                                 callbacks = [tb_callback])
            
score = model2.evaluate(X_test_n, y_test_n, verbose=1)

print(f'Loss: {score[0]}\nAccuracy: { round(score[1], 2)*100}%')
y_pred = np.argmax(model2.predict(X_test_n), axis = 1)


cm = metrics.confusion_matrix(y_test_n, y_pred)
f1_score1 = round(f1_score(y_test_n,y_pred,average='micro'),3)

print(f'F1 Score: {f1_score1*100}%\n')
print(f'Confusion Matrix:\n{cm}')
    
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
