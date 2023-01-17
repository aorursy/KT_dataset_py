# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Add, BatchNormalization

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils
train = np.genfromtxt('/kaggle/input/digit-recognizer/train.csv', delimiter=',',skip_header=1)

X_test = np.genfromtxt('/kaggle/input/digit-recognizer/test.csv', delimiter=',',skip_header=1)

print(train.shape, x_test.shape)
X_train =train[:,1:] 

y_train=train[:,0]

print(X_train.shape, y_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_test = X_test.reshape(X_test.shape[0], 28, 28)
print(X_train.shape, y_train.shape, X_test.shape)
print (X_train.shape)

from matplotlib import pyplot as plt

%matplotlib inline

plt.imshow(X_train[1])

print(y_train[1])
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
print(X_train.shape, y_train.shape, X_test.shape)
y_train[:10]

Y_train = np_utils.to_categorical(y_train, 10)
Y_train[:10]
from keras.layers import Activation

model = Sequential()

 

model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(28,28,1))) #26

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Convolution2D(16, 3, 3, activation='relu')) #24

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Convolution2D(8, 1, 1, activation='relu')) #24



model.add(MaxPooling2D(pool_size=(2, 2)))#12



model.add(Convolution2D(8, 3, 3, activation='relu'))#10

model.add(BatchNormalization())

model.add(Dropout(0.1))





model.add(Convolution2D(8, 3, 3, activation='relu'))#8

model.add(BatchNormalization())

model.add(Dropout(0.1))





model.add(Convolution2D(16, 3, 3, activation='relu'))#6

model.add(BatchNormalization())

model.add(Dropout(0.1))





model.add(Convolution2D(10, 6, 6))

model.add(BatchNormalization())

#model.add(Dropout(0.1))





model.add(Flatten())

model.add(Activation('softmax'))





model.summary()
from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):

  return round(lr * 1/(1 + 0.019 * epoch), 10)



model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])



model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, callbacks=[LearningRateScheduler(scheduler, verbose=1)])
y_pred = model.predict(X_test)
y_pred_val=y_pred.argmax(1)
y_pred_val
i=3992

plt.imshow(X_test[i][:,:,0])

print('Pred Label : ',y_pred_val[i])

X_test[i][:,:,0].shape
op_df = pd.DataFrame()

op_df['ImageId'] = np.arange(1,X_test.shape[0]+1)

op_df['Label'] = y_pred_val

op_df.to_csv (r'submission_labels.csv', index = None, header=True)
!ls /kaggle/working/