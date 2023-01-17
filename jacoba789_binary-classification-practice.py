import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
import numpy as np, matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import keras.layers 
import pandas as pd
import cv2
import matplotlib.image as img

tr_horse_dir='/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/horses/'         
tr_human_dir='/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/humans/'
test_horse_dir='/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/horses/'
test_human_dir='/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/humans/'
# put data in respective memory and label

def read_dat(dat_dir):
    tmp=[]
    for png in os.listdir(dat_dir):
        imgage=img.imread(dat_dir+png)
        imgage=cv2.cvtColor(imgage,cv2.COLOR_BGR2GRAY)
        tmp.append(imgage)
    return tmp
horse_train=read_dat(tr_horse_dir)
print('Number of horse training samples: ', len(horse_train))

human_train=read_dat(tr_human_dir)
print('Number of human training samples: ',len(human_train))

horse_val=read_dat(test_horse_dir)
print('Number of horse test samples: ',len(horse_val))

human_val=read_dat(test_human_dir)
print('Number of human test samples: ',len(human_val))


ax=[]
for j in range(8):
    ax.append(horse_train[j])
    plt.imshow(ax[j])
    plt.show()
#add the files together to make our training and test sets
all_train=np.concatenate((human_train, horse_train), axis = 0);
print(all_train.shape)
all_vali=np.concatenate((human_val,horse_val), axis = 0);
print(all_vali.shape)
x_data = np.concatenate((human_train, human_val, horse_train,horse_val), axis=0);
print(x_data.shape,'\nNote how theses are all 300x300 pixels')

# We create our classify data. 1 for human and 0 for horses. 
zero = np.zeros(len(horse_train) + len(horse_val)) # all horse images
one = np.ones(len(human_train) + len(human_val))   # all human images
print("Number of humans images :", one.size)
print("Number of horses images :", zero.size)
# Target data
y = np.concatenate((one, zero), axis= 0).reshape(-1,1)#the -1 infers the number of rows from the given number of colomns
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.3, random_state = 42)
print(y_train.shape)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2], 1)

x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2], 1)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 16
epochs = 5

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]
input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)



# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# Plotting our loss charts
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
