import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



import tensorflow as tf

from keras.datasets import cifar10

import numpy as np

import pandas as pd 

# loading dateset

(x_train, y_train),(x_test,y_test) = cifar10.load_data() 
# Checking loaded data 

print('Total number of Images in the Dataset:', len(x_train) + len(x_test))

print('Number of train images:', len(x_train))

print('Number of test images:', len(x_test))

print('Shape of training dataset:',x_train.shape)

print('Shape of testing dataset:',x_test.shape)

      
# This piece of code shows a random images and labels for given set of inputs 



def showImages(num_row,num_col,X,Y):

    import matplotlib.pyplot as plt

    %matplotlib inline



    from sklearn.utils import shuffle

    (X_rand, Y_rand) = shuffle(X, Y)

    

    fig, axes = plt.subplots(num_row,num_col,figsize = (12,12))

    axes = axes.ravel()

    for i in range(0, num_row*num_col):

        axes[i].imshow(X_rand[i])

        axes[i].set_title("{}".format(labels[Y_rand.item(i)]))

        axes[i].axis('off')

        plt.subplots_adjust(wspace =1)

        

    return

    
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
num_row = 10

num_col = 10

showImages(num_row,num_col,X =x_train,Y = y_train)

# Data visulization with seaborn  

# class count in train and test data



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





figure, axis = plt.subplots( 1, 2 , figsize = (18,6))

sns.countplot(y_train.ravel(),ax = axis[0])

axis[0].set_title('Distribution of training data')

axis[0].set_xlabel('Classes')

# Count plot for testing set

sns.countplot(y_test.ravel(), ax=axis[1])

axis[1].set_title('Distribution of Testing data')

axis[1].set_xlabel('Classes')

plt.show()
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train = x_train/ 255

x_test = x_test/255

#keras.utils.to_cotegorical(y, num_of_classes) convert the label 

#data into (m, num_of class) array from (m, 1) input 

import keras



y_train_one_hot = keras.utils.to_categorical(y_train, 10) #as classes = 10

y_test_one_hot = keras.utils.to_categorical(y_test, 10)
print('The one hot label is: ', y_train_one_hot[1])

print(y_train_one_hot.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
#Building model computational graph

model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = (32,32,3)))

model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.8))

model.add(Dense(10, activation = 'softmax'))

#Description about parameters and layers

model.summary()
# Flow chart of the model 



from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
#compiling model with loss, opt, metrics

loss = 'categorical_crossentropy'

opt = tf.keras.optimizers.Adam(learning_rate=0.0009,beta_1=0.9, beta_2=0.999,epsilon=1e-07)

metrics = ['accuracy']



model.compile(loss = loss, optimizer = opt, metrics = metrics)
# fitting the model for training dataset

hist = model.fit(x_train, y_train_one_hot, batch_size = 64 , epochs = 100, validation_split = 0.2)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
model.evaluate(x_test, y_test_one_hot)[1]