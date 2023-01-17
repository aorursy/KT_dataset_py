import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="ticks")

%matplotlib inline

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten ,Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

train_data = pd.read_csv("../input/digit-recognizer/train.csv")

test_data = pd.read_csv("../input/digit-recognizer/test.csv")



print(f'Training Data size is : {train_data.shape}')

print(f'Test Data size is : {test_data.shape}')
train_data.head()
test_data.head()
X = train_data.drop(['label'], axis=1, inplace=False)

y = train_data['label']



print('X shape is ' , X.shape)

print('y shape is ' , y.shape)
plt.figure(figsize=(12,10))

plt.style.use('ggplot')

for i in  range(20)  :

    plt.subplot(4,5,i+1)

    plt.imshow(X.values[ np.random.randint(1,X.shape[0])].reshape(28,28))
y.value_counts()
plt.figure(figsize=(12,12))

plt.pie(y.value_counts(),labels=list(y.value_counts().index),autopct ='%1.2f%%' ,

        labeldistance = 1.1,explode = [0.05 for i in range(len(y.value_counts()))] )

plt.show()

X = X / 255.0

test_data = test_data / 255.0
X.shape
X = X.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
X.shape
test_data.shape
ohe  = OneHotEncoder()

y = np.array(y)

y = y.reshape(len(y), 1)

ohe.fit(y)

y = ohe.transform(y).toarray()
y.shape
X_part, X_cv, y_part, y_cv = train_test_split(X, y, test_size=0.15, random_state=44, shuffle =True)



print('X_train shape is ' , X_part.shape)

print('X_test shape is ' , X_cv.shape)

print('y_train shape is ' , y_part.shape)

print('y_test shape is ' , y_cv.shape)
X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=0.25, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
KerasModel = keras.models.Sequential([

        keras.layers.Conv2D(filters = 32, kernel_size = (3,3),  activation = tf.nn.relu , padding = 'same'),

        keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='valid'),

        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(filters=32, kernel_size=(2,2),activation = tf.nn.relu , padding='same'),

        keras.layers.MaxPool2D(),

        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.5),        

        keras.layers.Flatten(),    

        keras.layers.Dropout(0.5),        

        keras.layers.Dense(64),    

        keras.layers.Dropout(0.3),            

        keras.layers.Dense(units= 10,activation = tf.nn.softmax ),                



    ])
KerasModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
KerasModel.fit(X_train,y_train,validation_data=(X_cv, y_cv),epochs=8,batch_size=64,verbose=1)
KerasModel.summary()
ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)



print('Test Loss is {}'.format(ModelLoss))

print('Test Accuracy is {}'.format(ModelAccuracy ))
y_pred = KerasModel.predict(X_test)



print('Prediction Shape is {}'.format(y_pred.shape))
for i in list(np.random.randint(0,len(X_test) ,size= 20)) : 

    print(f'for sample  {i}  the predicted value is   {np.argmax(y_pred[i])}   , while the actual letter is {np.argmax(y_test[i])}')

    if np.argmax(y_pred[i]) != np.argmax(y_test[i]) : 

        print('==============================')

        print('Found mismatch . . ')

        plt.figure(figsize=(5,5))

        plt.style.use('ggplot')

        plt.imshow(X_test[i].reshape(28,28))

        plt.show()

        print('==============================')
FinalResults = KerasModel.predict(test_data)

FinalResults = pd.Series(np.argmax(FinalResults,axis = 1) ,name="Label")



FileSubmission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),FinalResults],axis = 1)

FileSubmission.to_csv("sample_submission.csv",index=False)