# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from keras.utils import to_categorical



train_df=pd.read_csv('../input/digit-recognizer/train.csv')

print(train_df.shape)

test_df=pd.read_csv('../input/digit-recognizer/test.csv')

print(train_df.shape)

print('*** check *** aaa ***')

#

#
X_train = (train_df.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train_df.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test_df.values.astype('float32')

#

np.random.seed(666)

#

y_train = to_categorical(y_train)

 

# 学習データのフォーマットを修正

img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')

X_train /= 255

#



print('*** check *** bbb ***')
print('*** check *** ccc ***')

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

 

batch_size = 128

# 学習回数

# epochs = 3

# epochs = 5

# epochs = 7

epochs = 9

print('epochs = %d' % epochs)

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

 

# 最終的に出力される分類数 0~9 の10通り

num_classes = y_train.shape[1]

 

 

# モデルを作成

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

 

# 学習のためのモデルを設定

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])

 

 

# 学習

model.fit(X_train, y_train,

           batch_size=batch_size,

           epochs=epochs,

           verbose=1)

print('*** check *** fff ***')
print('*** check *** ggg ***')

# 学習データにフォーマットを合わせる

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_test = X_test.astype('float32')

X_test /= 255

 

predictions = model.predict_classes(X_test, verbose=0)

 

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("submission5.csv", index=False, header=True)

print('*** check *** kkk ***')
print('*** check *** lll ***')

nmax = len(predictions)

print('nmax = ',nmax)

icount = 0

for it in range(nmax):

    if predictions[it] != 1:

        print(it,predictions[it])

        icount += 1

    if 10 < icount:

        break

#

print('icount = ',icount)

print('*** check *** kkk ***')