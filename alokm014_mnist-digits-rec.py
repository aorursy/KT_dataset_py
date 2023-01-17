# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# importing the keras libraries and packages
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
# Helper libraries
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
# preprocessing the data
def get_features_labels(df):
    features=df.values[:,1:]/255
    labels=df['label'].values
    return features, labels
train_features, train_labels= get_features_labels(train_data)
test_features= test_data.values[:,0:]
# convert labels to one hot encoding
train_labels=tf.keras.utils.to_categorical(train_labels)
train_features.shape
#initialize ANN
classifier=Sequential()
# hidden layers
classifier.add(Dense(units=397,activation='relu', kernel_initializer='uniform',input_shape=(784,)))
classifier.add(Dense(units=397,kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=397,kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=397,kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=397,kernel_initializer='uniform', activation='relu'))
# output layer
classifier.add(Dense(units=10, kernel_initializer='uniform',activation='softmax'))
# compiling ANN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# fitting the ANN to the training set
EPOCHS=4
BATCH_SIZE=150
classifier.fit(train_features,train_labels,batch_size=BATCH_SIZE,epochs=EPOCHS)
predictions=classifier.predict(test_features)
np.argmax(predictions[0])
class_names=[0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(10,10))
for i in range(100):
    pred_label = np.argmax(predictions[i])
    pred_name = class_names[pred_label]
    test_array_data=np.reshape(test_data[test_data.columns[0:]].iloc[i].values,(28,28))
    plt.subplot(10,10,i+1)
    plt.title(pred_name)
    plt.imshow(test_array_data,'gray')
plt.tight_layout() 
