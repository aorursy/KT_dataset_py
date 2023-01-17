from keras.utils import to_categorical

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,accuracy_score

from keras.models import Sequential

from keras.layers import Dense

from keras.regularizers import l2
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test= pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
#Define the target and the features

x_train = train.drop(columns=['label'])

y_train = train.label



x_test = test.drop(columns=['label'])

y_test = test.label
#Print the shape of target

y_train.shape , y_test.shape
#Print the number of class in the train target

list(y_train.unique())
#Print the number of class in the test target

list(y_test.unique())
x_train.max().sort_values().tail(1)
x_test.max().sort_values().tail(1)
#The max number is 255, so lets divide the target by this number

x_train = x_train/255

x_test = x_test/255
x_train = x_train.astype(float) 

x_test = x_test.astype(float)
#Finally, lets get dummies of the target using the function presents in keras

from keras.utils import to_categorical



y_train = to_categorical(y_train,10) #10 levels of image

y_test =  to_categorical(y_test,10) #10 levels of image
#Starting a neural network

modelo = Sequential()



#Input the first layer in model with 50 neurals and the activation function will be Relu

modelo.add(Dense(50 #number os neurals

                ,activation = 'relu' #activation function

                ,input_shape = (784,) #Number of features in dataframe, let's pay attention, because keras need to receive a tuple, its the reason of (784,0)

                ,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)

                ))



#Input the second layer in model with 50 neurals and the same activation function

modelo.add(Dense(30 

                ,activation = 'relu' 

                ,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01) 

                ))



#Input the third layer in model with 20 neurals and the same activation function

modelo.add(Dense(20

                ,activation = 'relu' 

                ,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01) 

                ))



#The final layes will be with 10 neurals because we have 10 class in this dataframe

#The activation function is softmax because its will be normalize the output of neurals and it will be easier identify the probably of each class

modelo.add(Dense(10 #numero de classs

                ,activation = 'softmax' #Vai normaliza as probabilidades por exponencial

                ))



#Finally, lets see the summary of the model and see how many parameters its have.

modelo.summary()
#I use the croos entropy, because its penalty high error for bad clasification and a loss error for good erro, its a commum metric using in classification problems

modelo.compile(

                loss='categorical_crossentropy' 

               ,optimizer='adam' 

               ,metrics=['accuracy'] )
history = modelo.fit(x_train,y_train

         ,epochs=30 #number of times that model will through in train 

         ,batch_size = 128 #number of rows that will be consider to update the weights of layers

         ,verbose = 1

         ,validation_data=(x_test,y_test)

         )
#Print loss and accuracy

modelo.evaluate(x_test,y_test,verbose = 0)
plt.subplots(figsize=(13, 8))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.subplots(figsize=(13, 8))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
#Predict the x_test

p = modelo.predict(x_test)

p = (p > 0.5)

print('ACC: %.3f%%' % (accuracy_score(y_test, p)*100))

print('---------')

print(classification_report(y_test, p))
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(50,30,20) #define 3 layers with 50, 30 and 20 neurals

                      , batch_size=32 #Define the same bacth_size

                      , solver = 'adam' #Define the optimization

                      ,activation='relu' #Activation function

                      , max_iter=30 #Number of epochs

                      ,verbose=1

                      , random_state=42)



model.fit(x_train, y_train)
#Print the accuracy

print('Accuracy:', model.score(x_test, y_test))
plt.rcParams['figure.figsize'] = 10, 10



plt.plot(list(range(len(model.loss_curve_))), model.loss_curve_)
#Predict the x_test

p = model.predict(x_test)

p = (p > 0.5)

print('ACC: %.3f%%' % (accuracy_score(y_test, p)*100))

print('---------')

print(classification_report(y_test, p))