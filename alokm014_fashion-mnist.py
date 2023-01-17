# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# importing the keras libraries and packages
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
# Helper libraries
import matplotlib.pyplot as plt
# File path
train_data_file= "../input/fashion-mnist_train.csv"
test_data_file= "../input/fashion-mnist_test.csv"
# Read CSV file for Training and Testing
train_data=pd.read_csv(train_data_file)
test_data=pd.read_csv(test_data_file)
train_data.head()
# plot testing data and their labels
plt.figure(figsize=(10,10))
for i in range(25):
    test_array_data=np.reshape(test_data[test_data.columns[1:]].iloc[i].values/255,(28,28))
    plt.subplot(5,5,i+1)
    plt.title('Lables{}'.format(test_data['label'].iloc[i]))
    plt.imshow(test_array_data,'gray')
plt.tight_layout() 
# initialising the ANN
classifier=Sequential()
# hidden layer or fully connected layer
classifier.add(Dense(units=397,kernel_initializer='uniform', activation='relu',input_shape=(784,)))
classifier.add(Dense(units=397,kernel_initializer='uniform', activation='relu'))
# output layer
classifier.add(Dense(units=10, kernel_initializer='uniform',activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
def get_features_labels(df):
    features=df.values[:,1:]/255
    labels=df['label'].values
    return features, labels
train_features, train_labels= get_features_labels(train_data)
test_features, test_labels= get_features_labels(test_data)
train_labels.shape
train_labels=tf.keras.utils.to_categorical(train_labels)
test_labels=tf.keras.utils.to_categorical(test_labels)
train_labels.shape
EPOCHS=2
BATCH_SIZE=128
classifier.fit(train_features,train_labels,batch_size=BATCH_SIZE,epochs=EPOCHS)
predictions=classifier.predict(test_features)
test_loss, test_acc=classifier.evaluate(test_features,test_labels)
print(test_acc)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(100):
    pred_label = np.argmax(predictions[i])
    pred_name = class_names[pred_label]
    test_array_data=np.reshape(test_data[test_data.columns[1:]].iloc[i].values/255,(28,28))
    plt.subplot(10,10,i+1)
    plt.title(pred_name)
    plt.imshow(test_array_data,'gray')
plt.tight_layout() 
    
