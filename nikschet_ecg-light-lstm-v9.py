# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import math
import random
import pickle
import itertools

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

from sklearn.utils import shuffle

from scipy.signal import resample

import matplotlib.pyplot as plt

np.random.seed(42)

import pickle
from sklearn.preprocessing import OneHotEncoder




from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation# , Dropout
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import math
import random
import pickle
import itertools
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
np.random.seed(42)
import tensorflow as tf
import tensorflow.keras as keras


print(os.getcwd())
mit_test_data = pd.read_csv("../input/heartbeat/mitbih_test.csv", header=None)
mit_train_data = pd.read_csv("../input/heartbeat/mitbih_train.csv", header=None)
# There is a huge difference in the balanced of the classes.
# Better choose the resample technique more than the class weights for the algorithms.
from sklearn.utils import resample

df_1=mit_train_data[mit_train_data[187]==1]
df_2=mit_train_data[mit_train_data[187]==2]
df_3=mit_train_data[mit_train_data[187]==3]
df_4=mit_train_data[mit_train_data[187]==4]
df_0=(mit_train_data[mit_train_data[187]==0]).sample(n=20000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)

train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])


df_11=mit_test_data[mit_train_data[187]==1]
df_22=mit_test_data[mit_train_data[187]==2]
df_33=mit_test_data[mit_train_data[187]==3]
df_44=mit_test_data[mit_train_data[187]==4]
df_00=(mit_test_data[mit_train_data[187]==0]).sample(n=20000,random_state=42)

df_11_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)
df_22_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
df_33_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)
df_44_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)

test_df=pd.concat([df_00,df_11_upsample,df_22_upsample,df_33_upsample,df_44_upsample])


equilibre=train_df[187].value_counts()
print(equilibre)
print("ALL Train data")
print("Type\tCount")
print((mit_train_data[187]).value_counts())
print("-------------------------")
print("ALL Test data")
print("Type\tCount")
print((mit_test_data[187]).value_counts())

print("ALL Balanced Train data")
print("Type\tCount")
print((train_df[187]).value_counts())
print("-------------------------")
print("ALL Balanced Test data")
print("Type\tCount")
print((train_df[187]).value_counts())
#One hot encoding for categorical target
#Since we will be using neural networks for our classification model, 
#our output classes need to be turned into a numerical representation. We use one hot encoding (from sklearn package) to do this.



#train_target = mit_train_data[187]
#train_target = train_target.values.reshape(87554,1)
train_target = train_df[187]
train_target = train_target.values.reshape(100000,1)




#one hot encode train_target

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
# TODO: create a OneHotEncoder object, and fit it to all of X

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(train_target)

# 3. Transform
onehotlabels = enc.transform(train_target).toarray()
onehotlabels.shape

target = onehotlabels
#remove ground truth labels from training df
#train/test split


from sklearn.model_selection import train_test_split

#X = mit_train_data
X = train_df
X = X.drop(axis=1,columns=187)

X_train, X_valid, Y_train, Y_valid = train_test_split(X,target, test_size = 0.25, random_state = 36)
X_train = np.asarray(X_train)
X_valid = np.asarray(X_valid)
Y_train = np.asarray(Y_train)
Y_valid = np.asarray(Y_valid)

#X_train.reshape((1, 2403, 187))
X_train = np.expand_dims(X_train, axis=2)
X_valid = np.expand_dims(X_valid, axis=2)
print(X_train.shape)
print(Y_train.shape)
# 2,403 training heartbeats and 802 validation heartbeats 
# for a 75:25 train-test split. 
# MODEL 1 https://www.kaggle.com/freddycoder/heartbeat-categorization
# Separate features and targets

from keras.utils import to_categorical

print("--- X ---")
# X = mit_train_data.loc[:, mit_train_data.columns != 187]
X = train_df.loc[:, mit_train_data.columns != 187]
print(X.head())
print(X.info())

print("--- Y ---")
# y = mit_train_data.loc[:, mit_train_data.columns == 187]
y = train_df.loc[:, mit_train_data.columns == 187]
y = to_categorical(y)

print("--- testX ---")
#testX = mit_test_data.loc[:, mit_test_data.columns != 187]
testX = test_df.loc[:, mit_test_data.columns != 187]
print(testX.head())
print(testX.info())

print("--- testy ---")
#testy = mit_test_data.loc[:, mit_test_data.columns == 187]
testy = test_df.loc[:, mit_test_data.columns == 187]
testy = to_categorical(testy)
# Keras model to make prediction

#The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
#The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.
# softmax is used to categorize 

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(187,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10)

print("Evaluation: ")
mse, acc = model.evaluate(testX, testy)
print('mean_squared_error :', mse)
print('accuracy:', acc)
test= pd.read_csv("../input/arduino3a/AS3.txt", header=None)
test

plt.plot(test.iloc[100,:])
# NORMALIZING TEST DATA AMPLITUDE
from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
# prepare data for normalization
values = test.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
normalized = scaler.transform(values)

dfnormalized = pd.DataFrame(normalized)
dfnormalized.index = [x for x in range(1, len(dfnormalized.values)+1)]
plt.plot(dfnormalized.iloc[30,:])
# category= model.predict_classes(test) #not Normalized
category= model.predict_classes(dfnormalized) #Normalized
plt.plot(category)
np.mean(category)
test = pd.read_csv("../input/arduinorow1a/test44.txt", header=None)
#test = pd.read_csv("../input/arduinorow2a/marnelakis.txt", header=None)
#test = test.iloc[0,0:len(test.T)-1] # Remove last line cause it might be a Nan
test = pd.DataFrame(test)
test=test.T

# NORMALIZING TEST DATA AMPLITUDE
from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
# prepare data for normalization
values = test.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
normalized = scaler.transform(values)
normalized = pd.DataFrame(normalized) 
normalized

## TESTING FOR ONE X
#x=0
#normalized = pd.DataFrame(normalized.T) ## CAUTION!!! needs to run only once 
#normtest=normalized.iloc[0, 0+x:187+x] 
#normtest=pd.DataFrame(normtest)
#category = model.predict_classes(normtest.T)
#category
category= pd.DataFrame()
category=category.dropna()
lst_seq = np.arange(0,len(normalized.T)-190)
for x in lst_seq:
    normtest=normalized.iloc[0, 0+x:187+x] 
    normtest=pd.DataFrame(normtest)
    category[x] = model.predict_classes(normtest)
category
category
np.mean(category.T)
plt.plot(category.T)
category = pd.DataFrame(category)
temp1= category.iloc[0,:].value_counts()
print("Categories vs Value Count")
print(temp1)
print("Categories vs Frequency")
print(temp1/(len(category.T)))

target_train=train_df[187]
target_test=test_df[187]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)
X_train=train_df.iloc[:,:186].values
X_test=test_df.iloc[:,:186].values
#for i in range(len(X_train)):
#    X_train[i,:186]= add_gaussian_noise(X_train[i,:186])
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
def network(X_train,y_train,X_test,y_test):
    

    im_shape=(X_train.shape[1],1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1=BatchNormalization()(conv2_1)
    pool2=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
    conv3_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    pool3=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
    flatten=Flatten()(pool3)
    dense_end1 = Dense(64, activation='relu')(flatten)
    dense_end2 = Dense(32, activation='relu')(dense_end1)
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)
    
    
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history=model.fit(X_train, y_train,epochs=30,callbacks=callbacks, batch_size=32,validation_data=(X_test,y_test))
    model.load_weights('best_model.h5')
    return(model,history)
def evaluate_model(history,X_test,y_test,model):
    scores = model.evaluate((X_test),y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    target_names=['0','1','2','3','4']
    
    y_true=[]
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_test)
    prediction=np.argmax(prediction_proba,axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)
from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

model,history=network(X_train,y_train,X_test,y_test)
evaluate_model(history,X_test,y_test,model)
y_pred=model.predict(X_test)
df1 = pd.DataFrame()
category= pd.DataFrame()
category=category.dropna()
lst_seq = np.arange(0,len(normalized.T)-190)
for x in lst_seq:
    temp=normalized.iloc[0,0+x:186+x]
    temp=pd.DataFrame(temp) 
    temp=temp.values
    temp=temp.reshape(1,186,1)
    category=pd.DataFrame(model.predict(temp))
    df = pd.DataFrame(category)
    df1=df1.append(df)
    
category=df1
category=pd.DataFrame(category)
category
cat1=category[0].mean()
cat2=category[1].mean()
cat3=category[2].mean()
cat4=category[3].mean()
cat5=category[4].mean()
cat1

cat2
cat3
cat4
cat5

test = pd.read_csv("../input/arduinorow1a/test44.txt", header=None)
test = test.iloc[0,0:len(test.T)-1] # Remove last line cause it might be a Nan
test = pd.DataFrame(test)
# NORMALIZING TEST DATA AMPLITUDE
from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
# prepare data for normalization
values = test.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
normalized = scaler.transform(values)
normalized = pd.DataFrame(normalized)
normalized
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)
# MODEL 1 https://www.kaggle.com/freddycoder/heartbeat-categorization
# Separate features and targets

from keras.utils import to_categorical

print("--- X ---")
# X = mit_train_data.loc[:, mit_train_data.columns != 187]
X = train_df.loc[:, mit_train_data.columns != 187]
print(X.head())
print(X.info())

print("--- Y ---")
# y = mit_train_data.loc[:, mit_train_data.columns == 187]
y = train_df.loc[:, mit_train_data.columns == 187]
y = to_categorical(y)

print("--- testX ---")
#testX = mit_test_data.loc[:, mit_test_data.columns != 187]
testX = test_df.loc[:, mit_test_data.columns != 187]
print(testX.head())
print(testX.info())

print("--- testy ---")
#testy = mit_test_data.loc[:, mit_test_data.columns == 187]
testy = test_df.loc[:, mit_test_data.columns == 187]
testy = to_categorical(testy)
# create the model.
from keras.callbacks import History 
history = History()
embedding_vecor_length = 187
model = Sequential()
#model = Bidirectional(model)

model.add(Embedding(100000, embedding_vecor_length, input_length=187))
model.add(LSTM(187))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(X, y, validation_data=(testX, testy), epochs=3, batch_size=8)


#Dropout is a powerful technique for combating overfitting in your LSTM models 
#model = Sequential()
#model.add(Embedding(1000, embedding_vecor_length, input_length=187))
#model.add(LSTM(50, dropout=0.001, recurrent_dropout=0.001))
#model.add(Dense(5, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
#history = model.fit(X, y, validation_data=(testX, testy), epochs=50, batch_size=128)



## SAVE MODEL ##
# serialize model to JSON
model_json = model.to_json()
with open("1model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("1model.h5")
print("Saved model to disk")
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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

# The history for the validation dataset is labeled test by convention as it is indeed a test dataset for the model.
#The plots can provide an indication of useful things about the training of the model, such as:
#*It’s speed of convergence over epochs (slope).
#*Whether the model may have already converged (plateau of the line).
#*Whether the mode may be over-learning the training data (inflection for validation line)
y_pred = model.predict(testX, batch_size=1000)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

print(classification_report(testy.argmax(axis=1), y_pred.argmax(axis=1)))
category= pd.DataFrame()
category=category.dropna()
lst_seq = np.arange(0,len(normalized.T)-190)
for x in lst_seq:
    normtest=normalized.iloc[0, 0+x:187+x] 
    normtest=pd.DataFrame(normtest)
    category[x] = model.predict_classes(normtest.T)
category
np.mean(category.T)
plt.plot(category.T)
category = pd.DataFrame(category)
temp1= category.iloc[0,:].value_counts()
print("Categories vs Value Count")
print(temp1)
print("Categories vs Frequency")
print(temp1/(len(category.T)))
json_file = open("../working/model.json", 'r')
model_json = json_file.read() 
json_file.close()

from keras.models import model_from_json
model = model_from_json(model_json)
model.load_weights("../working/model.h5")

#model.compile(loss='binary_crossentropy', optimizer='adam')
#prediction = model.predict(x_test, batch_size=2048)[0].flatten()