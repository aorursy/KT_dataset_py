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

from keras.callbacks import ReduceLROnPlateau



train_data = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

test_data = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")



print(f'Training Data size is : {train_data.shape}')

print(f'Test Data size is : {test_data.shape}')
train_data.head()
test_data.head()
X = train_data.drop(['label'], axis=1, inplace=False)

y = train_data['label']



print('X shape is ' , X.shape)

print('y shape is ' , y.shape)
X_test = test_data.drop(['label'], axis=1, inplace=False)

y_test = test_data['label']



print('X shape is ' , X_test.shape)

print('y shape is ' , y_test.shape)
plt.figure(figsize=(12,10))

plt.style.use('ggplot')

for i in  range(20)  :

    plt.subplot(4,5,i+1)

    plt.imshow(X.values[ np.random.randint(1,X.shape[0])].reshape(28,28) , cmap='gray')

    
y.value_counts()
y_test.value_counts()
plt.figure(figsize=(12,12))

plt.pie(y.value_counts(),labels=list(y.value_counts().index),autopct ='%1.2f%%' ,

        labeldistance = 1.1,explode = [0.05 for i in range(len(y.value_counts()))] )

plt.show()

X = X / 255.0

X_test = X_test / 255.0
X.shape
X_test.shape
X = X.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
X.shape
X_test.shape
ohe  = OneHotEncoder()

y = np.array(y)

y = y.reshape(len(y), 1)

ohe.fit(y)

y = ohe.transform(y).toarray()
y.shape
ohe  = OneHotEncoder()

y_test = np.array(y_test)

y_test = y_test.reshape(len(y_test), 1)

ohe.fit(y_test)

y_test = ohe.transform(y_test).toarray()
y_test.shape
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.15, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_cv.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_cv.shape)
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
epochs_number = 200

hist = KerasModel.fit(X_train,y_train,validation_data=(X_cv, y_cv),epochs=epochs_number,batch_size=64,verbose=1)
score = KerasModel.evaluate(X_test, y_test, verbose=0)

score
KerasModel.summary()
ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)



print('Test Loss is {}'.format(ModelLoss))

print('Test Accuracy is {}'.format(ModelAccuracy ))
ModelAcc = hist.history['acc']

ValAcc = hist.history['val_acc']

LossValue = hist.history['loss']

ValLoss = hist.history['val_loss']

epochs = range(len(ModelAcc))
plt.plot(range(1,epochs_number+1),ModelAcc, 'ro', label='Accuracy of Training ')

plt.plot(range(1,epochs_number+1), ValAcc, 'r', label='Accuracy of Validation')

plt.title('Training Vs Validation Accuracy')

plt.legend()

plt.figure()

plt.plot(range(1,epochs_number+1), LossValue, 'ro', label='Loss of Training ')

plt.plot(range(1,epochs_number+1), ValLoss, 'r', label='Loss of Validation')

plt.title('Training Vs Validation loss')

plt.legend()

plt.show()
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