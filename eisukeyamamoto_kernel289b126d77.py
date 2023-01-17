# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(train.shape)

 

test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(test.shape)

 

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')

 

train.head()

 

np.random.seed(666)
import matplotlib.pyplot as plt

%matplotlib inline

 

X_train = X_train.reshape(X_train.shape[0], 28, 28)

 

fig = plt.figure(figsize=(9, 9))

fig.subplots_adjust(left=0, right=1, bottom=0, top=1.5, hspace=0.05, wspace=0.05)

index = 0 # 100*n

for i in range(0, 100):

    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])

    ax.imshow(X_train[i+index], cmap='gray')

    plt.title(str(i+index)+", "+str(y_train[i+index]));

import matplotlib.pyplot as plt

%matplotlib inline



X_test = X_test.reshape(X_test.shape[0], 28, 28)



fig = plt.figure(figsize=(9, 9))

fig.subplots_adjust(left=0, right=1, bottom=0, top=1.5, hspace=0.05, wspace=0.05)

index = 900

list1 = [57, 98, 128, 140, 508, 520, 582, 683, 706, 728, 702, 728, 759, 819, 831, 868, 888, 905, 934, 941]

for index, item in enumerate(list1):

    ax = fig.add_subplot(10, 10, index + 1, xticks=[], yticks=[])

    ax.imshow(X_test[item], cmap='gray')

    plt.title(item);
y_train= to_categorical(y_train,10)



# 学習データのフォーマットを修正

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')

X_train /= 255
batch_size = 128

# 学習回数

epochs = 3



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



# 最終的に出力される分類数 0~9 の10通り

num_classes = 10





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

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])

 

# 学習

model.fit(X_train, y_train,

           batch_size=batch_size,

           epochs=epochs,

           verbose=1)
# 学習データにフォーマットを合わせる

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_test = X_test.astype('float32')

X_test /= 255



predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)