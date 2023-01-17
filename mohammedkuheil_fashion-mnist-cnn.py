# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv',sep=',')

test_df = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv', sep = ',')
train_df.head()
train_data = np.array(train_df, dtype = 'float32')

test_data = np.array(test_df, dtype='float32')
x_train = train_data[:,1:]/255



y_train = train_data[:,0]



x_test= test_data[:,1:]/255



y_test=test_data[:,0]
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(16, 10))

for i in range(36):

    plt.subplot(6, 6, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i].reshape((28,28)))

    label_index = int(y_train[i])

    plt.title(class_names[label_index])

plt.show()
W_grid = 15

L_grid = 15



fig, axes = plt.subplots(L_grid, W_grid, figsize = (16,16))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_train = len(train_data) # get the length of the train dataset



# Select a random number from 0 to n_train

for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 



    # Select a random number

    index = np.random.randint(0, n_train)

    # read and display an image with the selected index    

    axes[i].imshow( train_data[index,1:].reshape((28,28)) )

    labelindex = int(train_data[index,0])

    axes[i].set_title(class_names[labelindex], fontsize = 9)

    axes[i].axis('off')



plt.subplots_adjust(hspace=0.3)
image_rows = 28

image_cols = 28

batch_size = 4096

image_shape = (image_rows,image_cols,1) 
x_train = x_train.reshape(x_train.shape[0],*image_shape)

x_test = x_test.reshape(x_test.shape[0],*image_shape)

x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)
model = Sequential([

    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),

    MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14

    Dropout(0.2),

    Flatten(), # flatten out the layers

    Dense(32,activation='relu'),

    Dense(10,activation = 'softmax')

    

])
model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
earlystop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
history = model.fit(

    x_train,

    y_train,

    batch_size=4096,

    epochs=75,

    verbose=1,

    validation_data=(x_validate,y_validate),callbacks=[earlystop]

)
plt.figure(figsize=(10, 10))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title('Train - Accuracy')
score = model.evaluate(x_test,y_test,verbose=0)

print('Test Loss : {:.4f}'.format(score[0]))

print('Test Accuracy : {:.4f}'.format(score[1]))
import matplotlib.pyplot as plt

%matplotlib inline

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')

plt.title('Training and Validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
num_classes = 10

#Get the predictions for the test data

predicted_classes = model.predict_classes(x_test)

#Get the indices to be plotted

y_true = test_df.iloc[:, 0]

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_true, predicted_classes, target_names=target_names))
L = 5

W = 5

fig, axes = plt.subplots(L, W, figsize = (12,12))

axes = axes.ravel()



for i in np.arange(0, L * W):  

    axes[i].imshow(x_test[i].reshape(28,28))

    axes[i].set_title(f"Prediction Class = {predicted_classes[i]:0.1f}\n Original Class = {y_test[i]:0.1f}")

    axes[i].axis('off')



plt.subplots_adjust(wspace=0.5)