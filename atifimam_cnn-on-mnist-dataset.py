import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



from keras.utils import plot_model

from keras.layers.normalization import BatchNormalization

from keras import losses

from IPython.display import Image

K.tensorflow_backend.set_image_dim_ordering('tf')



# Defining batch_size ,no of classes and no of epochs

batch_size=128

num_classes=10

epochs =12
# Input data dimensions

img_rows,img_cols=28,28
# Splitting the dataset into train and test

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
if K.image_data_format()=='channel_first':

    X_train=X_train.reshape(X_train.shape[0],1,img_rows,img_cols)

    X_test=X_test.reshape(X_test.shape[0],1,img_rows,img_cols)

    input_shape=(1,img_rows,img_cols)

else:

    X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)

    X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)

    input_shape=(img_rows,img_cols,1)
input_shape
# Normalizing

X_train=X_train.astype('float32')

X_test=X_test.astype('float32')

X_train=X_train/255

X_test=X_test/255
print (X_train.shape,X_train.shape[0],X_test.shape[0])
# Convert class vectors to binary class matrices

Y_train=keras.utils.to_categorical(Y_train,num_classes)

Y_test=keras.utils.to_categorical(Y_test,num_classes)
# Defining dynamic plot

%matplotlib notebook

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np 

import time

# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4 

# https://stackoverflow.com/a/14434334 # this function is used to update the plots for each epoch and error

def plt_dynamic(x, vy, ty, ax, colors=['b']):

  ax.plot(x, vy, 'b', label="Validation Loss")

  ax.plot(x, ty, 'r', label="Train Loss")

  plt.legend()

  plt.grid()

  fig.canvas.draw()

!apt install graphviz

!pip install pydot pydot-ng
def two_conv_layer(kernel_size, activation, padding, maxpooling,

                      dropout, strides, kernel_initializer, optimizer, batch_norm):

    model=Sequential()

    model.add(Conv2D(8,kernel_size=(3,3),activation=activation,padding=padding,input_shape=input_shape,

                     kernel_initializer=kernel_initializer))

    if maxpooling=='Y':

        model.add(MaxPooling2D(pool_size=(2,2),strides=strides))

    if dropout=='Y':

        model.add(Dropout(0.2))

    model.add(Conv2D(16, kernel_size=(3, 3), activation=activation, padding=padding,

                     kernel_initializer=kernel_initializer))

    if dropout=='Y':

        model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(120,activation=activation))

    if batch_norm=='Y':

        model.add(BatchNormalization())

    if dropout=='Y':

        model.add(Dropout(0.5))

    model.add(Dense(80,activation=activation))

    if batch_norm=='Y':

        model.add(BatchNormalization())

    if dropout=='Y':

        model.add(Dropout(0.5))

    model.add(Dense(num_classes,activation='softmax'))

    """Compining

    the

    model

    """

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    history=model.fit(X_train,Y_train,batch_size=batch_size,

                      epochs=epochs,verbose=1,validation_data=(X_test,Y_test))

    score_train = model.evaluate(X_train, Y_train, verbose=0)

    # test accuracy

    score_test = model.evaluate(X_test, Y_test, verbose=0)

    return model,history,score_train,score_test



model_2,history,train_score,test_score =two_conv_layer(kernel_size=(3,3), activation='relu', padding='valid', maxpooling='N',

                      dropout='N', strides=2, kernel_initializer='he_normal', optimizer='adam', batch_norm='N')


plot_model(model_2, show_shapes=True, show_layer_names=True, to_file='model_2.png')

Image(filename='model_2.png')
print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model_2,history,train_score,test_score =two_conv_layer(kernel_size=(3,3), activation='relu', padding='same', maxpooling='Y',

                      dropout='Y', strides=2, kernel_initializer='he_normal', optimizer='adam', batch_norm='Y')
plot_model(model_2, show_shapes=True, show_layer_names=True, to_file='model_2.png')

Image(filename='model_2.png')
print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model_2,history,train_score,test_score =two_conv_layer(kernel_size=(3,3), activation='relu', padding='same', maxpooling='Y',

                      dropout='Y', strides=2, kernel_initializer='he_normal', optimizer='adagrad', batch_norm='Y')
plot_model(model_2, show_shapes=True, show_layer_names=True, to_file='model_2.png')

Image(filename='model_2.png')
print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
def three_conv_layers(kernel_size, activation, padding, maxpooling,

                      dropout, strides, kernel_initializer, optimizer, batch_norm):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=kernel_size, activation=activation, padding=padding,

                     input_shape=input_shape, kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides))

    if dropout == 'Y':

        model.add(Dropout(0.20))

    model.add(Conv2D(64, kernel_size=kernel_size, activation=activation,

                     padding=padding,

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides))

    if dropout == 'Y':

        model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=kernel_size, activation=activation,

                     padding=padding,

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides))



    if dropout == "Y":

        model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation=activation))

    if batch_norm == 'Y':

        model.add(BatchNormalization())

    if dropout == 'Y':

        model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))



    """

    Compining the model

    """

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    history=model.fit(X_train,Y_train,batch_size=batch_size,

                      epochs=epochs,verbose=1,validation_data=(X_test,Y_test))

    score_train = model.evaluate(X_train, Y_train, verbose=0)

    # test accuracy

    score_test = model.evaluate(X_test, Y_test, verbose=0)

    



 

    return model,history,score_train,score_test
model_3,history,train_score,test_score =three_conv_layers(kernel_size=(3,3), activation='relu', padding='valid', maxpooling='N', dropout='N', strides=2, kernel_initializer='he_normal', optimizer='adam', batch_norm='N')


plot_model(model_3, show_shapes=True, show_layer_names=True, to_file='model_3.png')

Image(filename='model_3.png')
print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model_3,history,train_score,test_score =three_conv_layers(kernel_size=(3,3), activation='relu', padding='same', maxpooling='Y',

                      dropout='Y', strides=2, kernel_initializer='he_normal', optimizer='adam', batch_norm='Y')


plot_model(model_3, show_shapes=True, show_layer_names=True, to_file='model_3.png')

Image(filename='model_3.png',retina=True)
print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model_3,history,train_score,test_score =three_conv_layers(kernel_size=(3,3), activation='relu', padding='same', maxpooling='Y',

                      dropout='Y', strides=2, kernel_initializer='he_normal', optimizer='adagrad', batch_norm='Y')


plot_model(model_3, show_shapes=True, show_layer_names=True, to_file='model_3.png')

Image(filename='model_3.png',retina=True)

print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
def five_conv_layers(kernel_size, activation, padding, maxpooling,

                      dropout, strides, kernel_initializer, optimizer, batch_norm):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=kernel_size, activation=activation, padding=padding,

                     input_shape=input_shape, kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides,dim_ordering="tf"))

    if dropout == 'Y':

        model.add(Dropout(0.20))

    model.add(Conv2D(64, kernel_size=kernel_size, activation=activation,

                     padding=padding,

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides,dim_ordering="tf"))

    if dropout == 'Y':

        model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=kernel_size, activation=activation,

                     padding=padding,

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides,dim_ordering="tf"))



    if dropout == "Y":

        model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=kernel_size, activation=activation,

                     padding=padding,

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides,dim_ordering="tf"))



    if dropout == "Y":

        model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=kernel_size, activation=activation,

                     padding=padding,

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides,dim_ordering="tf"))



    if dropout == "Y":

        model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation=activation))

    if batch_norm == 'Y':

        model.add(BatchNormalization())

    if dropout == 'Y':

        model.add(Dropout(0.5))

    model.add(Dense(128, activation=activation))

    if batch_norm == 'Y':

        model.add(BatchNormalization())

    if dropout == 'Y':

        model.add(Dropout(0.5))

    model.add(Dense(64, activation=activation))

    if batch_norm == 'Y':

        model.add(BatchNormalization())

    if dropout == 'Y':

        model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))



    """

    Compining the model

    """

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    history=model.fit(X_train,Y_train,batch_size=batch_size,

                      epochs=epochs,verbose=1,validation_data=(X_test,Y_test))

    score_train = model.evaluate(X_train, Y_train, verbose=0)

    score_test = model.evaluate(X_test, Y_test, verbose=0)



    return model,history,score_train,score_test
model_5,history,train_score,test_score =five_conv_layers(kernel_size=(3,3), activation='relu', padding='valid', maxpooling='N',

                      dropout='N', strides=2, kernel_initializer='he_normal', optimizer='adam', batch_norm='N')
plot_model(model_5, show_shapes=True, show_layer_names=True, to_file='model_5.png')

Image(filename='model_5.png',retina=True)
print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model_5,history,train_score,test_score =five_conv_layers(kernel_size=(3,3), activation='relu', padding='same', maxpooling='N',

                      dropout='Y', strides=2, kernel_initializer='he_normal', optimizer='adam', batch_norm='Y')
print('Train score:', train_score[0])

print('Train accuracy:', train_score[1]*100)

print('\n************************ *********************\n')

#test accuracy

print('Test score:', test_score[0])

print('Test accuracy:', test_score[1]*100)





# plot

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch');

ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,12+1)) 

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
from keras.initializers import he_normal,glorot_normal

def three_conv_layers( activation, optimizer):

    model = Sequential()

    maxpooling="Y"

    dropout="Y"

    strides=2

    batch_norm="Y"

    if activation == 'relu':

        kernel_initializer = he_normal(seed=None)

    if activation == 'sigmoid' or activation == 'tanh':

        kernel_initializer = glorot_normal(seed=None)

    model.add(Conv2D(32, kernel_size=(3,3), activation=activation, padding='same',

                     input_shape=input_shape, kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides))

    if dropout == 'Y':

        model.add(Dropout(0.20))

    model.add(Conv2D(64, kernel_size=(3,3), activation=activation,

                     padding='same',

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides))

    if dropout == 'Y':

        model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3,3), activation=activation,

                     padding='same',

                     kernel_initializer=kernel_initializer))

    if maxpooling == 'Y':

        model.add(MaxPooling2D(pool_size=(2, 2), strides=strides))



    if dropout == "Y":

        model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation=activation))

    if batch_norm == 'Y':

        model.add(BatchNormalization())

    if dropout == 'Y':

        model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))



    """

    Compining the model

    """

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
param_grid={

    'activation' :['sigmoid','relu'],

    'optimizer':['adam','sgd','adagrad','rmsprop']

}

nb_epoch=12



from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV



model = KerasClassifier(build_fn=three_conv_layers, epochs=nb_epoch, batch_size=batch_size, verbose=0)

# param_grid = dict(activ=activ)
grid = GridSearchCV(estimator=model, param_grid=param_grid,verbose=10,cv=3)
grid_result=grid.fit(X_train,Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from prettytable import PrettyTable

tb = PrettyTable()

tb.field_names= ("Conv layer", "Model", "Train Accuracy", "Test Accuracy")

tb.add_row(["2", "2Conv + ADAM + RELU+3 Hidden Layer",99.79,98.65])

tb.add_row(["2", "2Conv + ADAM + RELU+3 Hidden Layer+maxpooling+dropout+batch_normalization+padding",99.45,98.83])

tb.add_row(["2", "2Conv + Adagrad + RELU+3 Hidden Layer+maxpooling+dropout+batch_normalization+padding",98.94,98.59])



tb.add_row(["3", "3 Conv + ADAM + RELU +2hidden layer",99.92,98.8])

tb.add_row(["3", "3Conv + ADAM + RELU + batch_normalization+dropout+maxpooling",99.56,99.26])

tb.add_row(["3", "3Conv + adagrad + RELU + batch_normalization+dropout+maxpooling+padding",99.36,99.29])





tb.add_row(["5", "5 Conv + ADAM + RELU+4 hidden layer",99.89,99.06])

tb.add_row(["5", "5 conv+ ADAM + RELU + batch_normalization+dropout+padding",99.80,99.32])





print(tb.get_string(titles = "CNN Models - Observations"))