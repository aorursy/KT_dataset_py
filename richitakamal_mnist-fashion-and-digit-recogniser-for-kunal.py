import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data_digit = pd.read_csv("../input/digit-recognizer/train.csv")

test_data_digit = pd.read_csv('../input/digit-recognizer/test.csv')



train_data_fashion = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test_data_fashion = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train_data_digit.head()
test_data_fashion.head(10)
X_train_full_digit = train_data_digit.loc[:,"pixel0":"pixel783"]

y_train_full_digit = train_data_digit['label']
X_train_arr_digit = np.asarray(X_train_full_digit)
X_train_arr_digit
X_train_arr_digit = X_train_arr_digit.reshape(X_train_full_digit.shape[0],28,28)



print(X_train_arr_digit.shape)
#We'll do this with the y_train_full and test_data



#Y_TRAIN_FULL



y_train_arr_digit = train_data_digit.values[:,1]



#One hot encoding the y_train labels

from keras.utils.np_utils import to_categorical

y_train_arr = to_categorical(y_train_full_digit)



#X_TEST



X_test_arr_digit = test_data_digit.values[:,:]

X_test_arr_digit = X_test_arr_digit.reshape(test_data_digit.shape[0],28,28,1)



print(y_train_arr_digit)

print(X_test_arr_digit.shape)
import matplotlib.pyplot as plt

num = 43

img = X_train_arr_digit[num]

plt.imshow(img,cmap = 'gray')



print("Number is ",y_train_full_digit[num])
X_train_arr_digit = X_train_arr_digit/255

x_test_arr_digit = X_test_arr_digit/255
num = 43

img = X_train_arr_digit[num]

plt.imshow(img,cmap = 'gray')



print("Number is ",y_train_full_digit[num])
#First we will reshape the X_train_arr and y_train_arr by adding one more dimension, that will specify it's channel value.

#As this is Black and White, we will chose 1



X_train_arr_digit = X_train_arr_digit.reshape(X_train_arr_digit.shape[0],28,28,1)

X_test_arr_digit = X_test_arr_digit.reshape(X_test_arr_digit.shape[0],28,28,1)
import keras



from keras.models import  Sequential

from keras.layers.core import Dense, Flatten, Dropout

from keras.layers import Conv2D



BATCH_SIZE = 64

EPOCHS = 30

ROW_DIM = 28

COL_DIM = 28

NUM_CLASSES = 10 #10 numbers in total from 0 to 9
y = train_data_digit.label

out_y = keras.utils.to_categorical(y, 10)
from sklearn.model_selection import train_test_split



X_train,X_valid,y_train,y_valid = train_test_split(X_train_arr_digit,out_y,test_size = 0.1,random_state = 42)
model1 = Sequential()



model1.add(Conv2D(32,kernel_size = (3,3),input_shape = (28,28,1),activation = 'relu'))

model1.add(Conv2D(32,kernel_size = (3,3), activation = 'relu'))

model1.add(Conv2D(64,kernel_size = (3,3), activation = 'relu'))

model1.add(Flatten())



model1.add(Dense(128,activation = 'relu'))

model1.add(Dense(NUM_CLASSES,activation = 'softmax'))
model1.compile( loss = keras.losses.categorical_crossentropy,

              optimizer = 'adam',

              metrics = ['accuracy'])
model1.summary()
model1.fit(X_train,y_train,

          batch_size = 128,

          epochs = 10,

          validation_split = 0.2)
preds = model1.predict(X_test_arr_digit)
preds[2]
test_view = X_test_arr_digit



test_view = test_view.reshape(X_test_arr_digit.shape[0],28,28)



import cv2

import matplotlib.pyplot as plt



plt.imshow(test_view[2])
epochs = [1,2,3,4,5,6,7,8,9,10]

accuracy = [97.90,98.35,98.47,98.39,98.64,98.52,98.44,98.07,98.54,98.56]

google_accuracy = [66.72,77.03,79.91,81.43,82.57,83.25,83.77,84.21,84.55,84.92]



sns.lineplot(x = epochs, y = accuracy, marker = 'o', label = 'MNIST Digit')

sns.lineplot(x = epochs, y = google_accuracy, marker = 'o', color = 'red', label = 'MNIST Fashion')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()