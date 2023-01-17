import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import itertools
%matplotlib inline
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df  =pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
# Datset shape
df.shape
# let's check what different classes are present in our dataset

df['Class'].unique()
_ = sns.countplot(x='Class',data=df)
fig, ax = plt.subplots()
_ = sns.kdeplot(df['Class'] , color='gray',bw=0.15)
df.describe()
scaler = MinMaxScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
#what are the representation of each class in dataset, checking numbers

fraudlent_df = df[df['Class']==1]
non_fraudlent_df = df[df['Class']==0]

num_fraudlent_df = len(df[df['Class']==1])
num_non_fraudlent_df = len(df[df['Class']==0])

print(f'number of fraudlent transaction : {num_fraudlent_df}')
print(f'number of non- fraudlent transaction : {num_non_fraudlent_df}')
train_fraudlent_df = resample(fraudlent_df, 
                                 replace=False,    # sample without replacement
                                 n_samples=num_fraudlent_df-100,     #  taking 392 out of 492 in train , 80%
                                 random_state=123) # reproducible results
train_non_fraudlent_df = resample(non_fraudlent_df, 
                                 replace=False,    # sample without replacement
                                 n_samples=num_fraudlent_df-100,     # to match minority class, taking 392 out of 284315 in train
                                 random_state=123) # reproducible results
train_non_fraudlent_df.head()
train_fraudlent_df.head()
train_non_fraudlent_df.shape
train_fraudlent_df.shape
train_df = pd.concat([train_non_fraudlent_df, train_fraudlent_df]) # final training set
train_df = train_df.sample(frac=1)   # shuffling the whole train dataframe
train_df.head()
train_df_index = train_df.index
df.drop(train_df_index, inplace=True ) # after dropping the rows from train dataframe , the remaining will be used for testing
train_df.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
x_train = train_df.values[:,:-1]
y_train = train_df.values[:,-1]
x_train.shape
y_train.shape
x_test = df.values[:,:-1]
y_test  = df.values[:,-1]
x_test.shape
y_test.shape
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
predictions_knn = knn_model.predict(x_test)
knn_model.score(x_test,y_test)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
predictions_log = logreg.predict(x_test)
logreg.score(x_test,y_test)
dec_tree = DecisionTreeClassifier(random_state=0)
dec_tree.fit(x_train, y_train)
predictions_dt = dec_tree.predict(x_test)
dec_tree.score(x_test,y_test)
ran_for = RandomForestClassifier(random_state=0)
ran_for.fit(x_train, y_train)
predictions_rf = ran_for.predict(x_test)
ran_for.score(x_test,y_test)
gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train, y_train)
predictions_gb = gb.predict(x_test)
gb.score(x_test,y_test)
prediction_list = [predictions_knn, predictions_log,  predictions_dt, predictions_rf, predictions_gb ]
name = ['KNN','LOGISTIC REGRESSION', 'DECISION TREE', 'RANDOM FORREST', 'GRADIENT BOOST']  
def plot_confusion_matrix(cm,classes,normalize=False,
                          title='confusion_matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap= cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=30)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('confusion  matrix without normalization')
    print(cm)
    
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('true label')

    plt.xlabel('predicted label')
plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(3,2,i+1)
    cm = confusion_matrix(y_test, prediction_list[i])
    cm_plot_labels = ['Non-Fraudlent', 'Fraudlent']
    _ = plot_confusion_matrix(cm,cm_plot_labels,title=f'{name[i]}')
activation_fn           = 'relu'
num_hidden_unit_layer_1 = 30
num_hidden_unit_layer_2 = 32
num_output_unit         = 2
num_epoch               = 20
batch_size              = 10
learning_rate           = 0.001
nn=Sequential([
    Dense(30, input_shape=(30,), activation=activation_fn),
    Dense(32, activation=activation_fn),
    Dense(num_output_unit, activation='softmax')
])
nn.summary()
optimizer = Adam(lr=learning_rate)
nn.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history =nn.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=num_epoch, shuffle=True, verbose=2)
# summarize history for accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

score, acc = nn.evaluate(x_test, y_test,batch_size=100)
print(f'Test Score   : {score}')
print(f'Test Accuracy:   {acc}')
predict =nn.predict(x_test)
predictions = nn.predict_classes(x_test, batch_size=100, verbose=0)
cm = confusion_matrix(y_test, predictions)
cm_plot_labels = ['Non-Fraudlent', 'Fraudlent']
_ = plot_confusion_matrix(cm,cm_plot_labels,title='Confusion_matrix')
# nn.save('credit_card_fraud_detection.h5')