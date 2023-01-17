import numpy as np
import pandas as pd
import pickle
import os

print(os.listdir("../input/shape count"))
with open('../input/shape count/train_images_transparent.pkl', 'rb') as inputfile:
    x_train = pickle.load(inputfile)
    
with open('../input/shape count/test_images_transparent.pkl', 'rb') as inputfile:
    x_test = pickle.load(inputfile)
    
y_train = pd.read_csv('../input/shape count/train_labels.csv', header = None)

y_test = pd.read_csv('../input/shape count/test_labels.csv',header = None)

y_train = np.array(y_train)

y_test = np.array(y_test)
from matplotlib import pyplot as plt
import random

rand = random.randint(0,len(x_train))

print('This image has:')
print(str(y_train[rand][0]) + ' blue circles' )
print(str(y_train[rand][1]) + ' green squares' )
print(str(y_train[rand][2]) + ' red squares' )
plt.imshow(x_train[rand]);
from keras import backend as K

img_rows, img_cols = 100,100

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,3)
    input_shape = (img_rows, img_cols, 3)
    
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

model1 = Sequential()
model1.add(Flatten(input_shape=x_train.shape[1:4]))
model1.add(Dense(512, activation='relu'))
model1.add(Dense(y_train.shape[1], activation='relu'))

model2 = Sequential()
model2.add(Conv2D(filters = 32, kernel_size=(3,3), padding = 'same', input_shape=x_train.shape[1:4]))
model2.add(Conv2D(filters = 64, kernel_size=(3,3), padding = 'same'))
model2.add(MaxPooling2D())
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dense(y_train.shape[1], activation='relu'))
batch_size = 128
epochs = 5

model1.compile(loss='mean_squared_error',
              optimizer='adam')

model1.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

score1 = model1.evaluate(x_test, y_test, verbose=1)

model2.compile(loss='mean_squared_error',
              optimizer='adam')

model2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

score2 = model2.evaluate(x_test, y_test, verbose=1)

print('MLP loss:', score1)
print('CNN loss:', score2)
miscounts1 = []
miscounts2 = []

for i in range(0,len(x_test)-1):

    sample = x_test[i]
    sample = np.expand_dims(sample,axis=0)
    sample1 = np.round(model1.predict(sample))
    sample2 = np.round(model2.predict(sample))
    sample1 = np.abs(y_test[i] - sample1)
    sample2 = np.abs(y_test[i] - sample2)
    miscounts1.append(sample1)
    miscounts2.append(sample2)
    

totals = pd.DataFrame(pd.DataFrame(y_test).sum())
totals1 = pd.DataFrame(pd.DataFrame(np.array(miscounts1).squeeze()).sum())
totals2 = pd.DataFrame(pd.DataFrame(np.array(miscounts2).squeeze()).sum())
totals = pd.concat([totals,totals1,totals2],axis=1)


totals.columns = ['Test set totals', 'MLP miscounts', 'CNN miscounts']
totals.rename(index={0:'Blue Circles',1:'Green Squares', 2:'Red Squares'}, inplace=True)
totals.loc['Total'] = totals.sum()

print(totals)
