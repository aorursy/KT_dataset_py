import matplotlib.pyplot as plt
%matplotlib inline
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
df=pd.read_csv("../input/6peakmixednoise5-10-15db/6peakMixedNoise5_10_15db_4049.csv")



df=df.dropna()
df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
df = df.dropna()
df
df
target=df.iloc[:,0]

data=df.iloc[:,1:]
import gc
del df
gc.collect()
print(data.shape)
print("different class data counts",target.value_counts())
target.value_counts().plot(kind='bar', title='Count (Unbalanced Classes)');


from sklearn.utils import resample

from keras.utils import to_categorical
print("--- testX ---")
testX = pd.DataFrame(data)
print(testX.head())
print(testX.info())
X_test=testX

print("--- testy ---")
testy = pd.DataFrame(target)

testy = to_categorical(testy)
y_test=testy

testX = np.expand_dims(testX,2)
testX.shape

#y_pred = loaded_model.predict(testX, batch_size=1000)
from keras import backend as K
    
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
[X_train, X_test, y_train, y_test] = train_test_split(data, target, test_size=0.2, random_state=10, stratify=target)
from keras.utils import to_categorical

print("--- X ---")
X = pd.DataFrame(X_train)
X_train=X
print(X.head())
print(X.info())

print("--- Y ---")
y = pd.DataFrame(y_train)
y = to_categorical(y)
print(y.shape)
y_train=y

print("--- testX ---")
testX = pd.DataFrame(X_test)
print(testX.head())
print(testX.info())
X_test=testX

print("--- testy ---")
testy = pd.DataFrame(y_test)
testy = to_categorical(testy)
y_test=testy
print(X.shape)
print(y.shape)
print(testX.shape)
print(testy.shape)
from keras import backend as K
del data
del target
    
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
X = np.expand_dims(X,2)
testX = np.expand_dims(testX,2)
n_obs, feature, depth = X.shape
batch_size = 512


gc.collect()
def build_model():
    input_img = Input(shape=(feature, depth), name='ImageInput')
    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling1D(2, name='pool1')(x)
    
    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling1D(2, name='pool2')(x)
    
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    x = SeparableConv1D(256, 3, activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling1D(2, name='pool3')(x)
    x = Dropout(0.6, name='dropout0')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.6, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(14, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model
model =  build_model()

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc',f1])
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
history = model.fit(X, y, validation_split=0.2,epochs=100,batch_size=batch_size,shuffle=True,callbacks=[checkpointer])
print("Evaluation: ")
mse, acc, F1 = model.evaluate(testX, testy)
print('mean_squared_error :', mse)
print('accuracy:', acc)
print('F1:', F1)
#model.save('cnn-0.9801.h5')

model.save('6peakmodel.h5')
model.save_weights("Mixed6peakweight.h5")
print("Saved model01 to disk")

model_json = model.to_json()
with open("Mixed6peak.json", "w") as json_file:
    json_file.write(model_json)
import os
os.chdir(r'../working')
from IPython.display import FileLink
FileLink(r'Mixed6peakweight.h5')
model_yaml = model.to_yaml()
with open("Bal6peak.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
#model.save_weights("Bal6peakweight.h5")
#print("Saved model to disk")
y_pred = model.predict(testX, batch_size=1000)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

print(classification_report(testy.argmax(axis=1), y_pred.argmax(axis=1)))
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import itertools  
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(testy.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8, 8))
plot_confusion_matrix(cnf_matrix, classes=['N', 'AB','AFIB', 'AFL','B','BII', 'T', 'VFL','VT', 'SVTA', 'IVR', 'P', 'SBR', 'NOD' ],
                     title='Confusion matrix, with normalization')
plt.show()
sens = []

spec = []
acc = []

for each in range(0,14):
    match=sum(testy.argmax(axis=1)== y_pred.argmax(axis=1))
   
    sens.append( cnf_matrix[each, each] / sum( cnf_matrix[each, :]))
    spec.append((match -  cnf_matrix[each, each]) / ((match - cnf_matrix[each, each] + sum( cnf_matrix[:, each]) -  cnf_matrix[each, each])))
    
speci = pd.DataFrame(spec)
spec=speci[0]


macc=sens-spec
sens= np.array(sens)
sens=np.transpose(sens)
import pandas as pd 
  
# intialise data of lists. 
data = {'Sensitivity':sens, 'Specificty':spec,'MAcc':(sens+spec)/2} 
  
# Create DataFrame 
result = pd.DataFrame(data) 
  
# Print the output. 
result
