import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import random

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
fashion_train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

fashion_test  = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
print(fashion_train.shape)

print(fashion_test.shape)
training = np.array(fashion_train, dtype = 'float32')

testing = np.array(fashion_test, dtype = 'float32')
i = random.randint(1, 60000)

plt.imshow(training[i,1:].reshape(28,28))

label = training[i,0]

label
w_grid = 15                                                                 #0 T-shirt/top

l_grid = 15                                                                 #1 Trouser

fig, axes = plt.subplots(l_grid, w_grid, figsize = (17,17))                 #2 Pullover

axes = axes.ravel()                                                         #3 Dress

n_training = len(training)                                                  #4 Coat

for i in np.arange(0, w_grid*l_grid):                                       #5 Sandal

    index = np.random.randint(0, n_training)                                #6 Shirt

    axes[i].imshow(training[i,1:].reshape(28,28))                           #7 Sneaker

    axes[i].set_title(training[index,0], fontsize= 8)                       #8 Bag

    axes[i].axis('off')                                                     #9 Ankle boot



plt.subplots_adjust(hspace = 0.4)                                                                                
x_train = training[:,1:]/255

y_train = training[:,0]

x_test = testing[:,1:]/255

y_test = testing[:,0]



x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.2, random_state = 12345)
x_train = x_train.reshape(x_train.shape[0], *(28,28,1))

x_test = x_test.reshape(x_test.shape[0], *(28,28,1))

x_validate = x_validate.reshape(x_validate.shape[0], *(28,28,1))

print(x_train.shape)

print(x_test.shape)

print(x_validate.shape)
cnn_model = Sequential()

cnn_model.add(Conv2D(32,3,3, input_shape=(28,28,1), activation = 'relu'))

cnn_model.add(MaxPool2D(pool_size = (2,2)))





cnn_model.add(Flatten())

cnn_model.add(Dense(units = 1032, activation= 'relu'))

cnn_model.add(Dense(units = 10, activation= 'sigmoid'))

cnn_model.compile(loss= 'sparse_categorical_crossentropy', optimizer= Adam(lr = 0.001), metrics = ['accuracy'])
cnn_model.fit(x_train, y_train, epochs = 50, verbose = 1, batch_size=512, validation_data=(x_validate, y_validate))
evaluate = cnn_model.fit(x_test, y_test)

evaluate
predicted_Class = cnn_model.predict_classes(x_test)
cm = confusion_matrix(y_test, predicted_Class)

plt.figure(figsize= (16,16))

sns.heatmap(cm, annot = True)
w_grid = 8

l_grid = 8



fig, axes = plt.subplots(l_grid, w_grid, figsize = (17,17))



axes = axes.ravel()



n_training = len(training)



for i in np.arange(0, w_grid*l_grid):

    

    axes[i].imshow(x_test[i].reshape(28,28))

    axes[i].set_title('Predicted {:0.1f}\n True {:0.1f}'.format(predicted_Class[i], y_test[i]),fontsize= 8)

    axes[i].axis('off')

    

plt.subplots_adjust(hspace = 0.4)
num = 10



target_names = ['Class {}'.format(i) for i in range(num)]



print(classification_report(y_test, predicted_Class, target_names= target_names))