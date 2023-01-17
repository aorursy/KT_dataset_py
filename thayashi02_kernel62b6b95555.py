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
import os

import keras

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, LeakyReLU, Input,Dense, Dropout, Activation, Flatten

from keras.models import Sequential

from keras.utils import to_categorical





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# 学習データ、テストデータを読み込み

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').to_numpy()





train.head()

x_train = train.drop('label', axis=1).to_numpy()

y_train = train['label'].to_numpy()

y_train = to_categorical(y_train, num_classes = 10) 

print(x_train.shape, y_train.shape)





#データを 縦28 ×  横28 ×　１(RGBだと3, 今回は白黒なので1) の形に整形

x_train = x_train.reshape(-1, 28, 28, 1)

test = test.reshape(-1, 28, 28, 1)



x_train = x_train.astype('float32')

test = test.astype('float32')



# 正規化

x_train /= 255

test /= 255



opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

EPOCHS = 100



# トレーニングデータを使って学習####################################

# ニューラルネットワークの層構成(今回はCNNという深層学習の１つを使用)

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





# model.add(GlobalAveragePooling2D())

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))





# 学習に関する色々な設定

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



# 実際に学習を開始、epochsは学習回数を表す

history = model.fit(x_train, y_train, batch_size=64, epochs=EPOCHS, )

##############################################################





# テストデータを使って推論

preds = model.predict(test)



# 提出

df = pd.DataFrame(np.argmax(preds, axis=1), columns=['Label'])

df.insert(0, 'ImageId', df.index + 1)

df.to_csv('submission.csv', index=False)