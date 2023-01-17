# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#再現性の確保

import random as rn



os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(7)

rn.seed(7)
#正常画像の読み込み

import glob

from keras.preprocessing.image import load_img, img_to_array

from tqdm import tqdm



img_size = (220, 220)

img_array_list = []

cls_list = []



#ラベル値　正常：0、欠陥：1

dir = '../input/1056lab-defect-detection-extra/train/Class[1-6]'

img_list = glob.glob(dir + '/*.png')

img_list.sort()



for i in tqdm(img_list, desc="Loading Normal.png files"):

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255.0

    img_array_list.append(img_array)

    cls_list.append(0)
#欠陥画像の読み込み

from keras.preprocessing.image import random_rotation, random_shift, random_zoom

img_list1 = glob.glob(dir + '_def/*.png')

img_list1.sort()

for i in tqdm(img_list1, desc="Loading Defect.png files"):

    img1 = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array1 = img_to_array(img1) / 255.0

    img_array_list.append(img_array1)

    cls_list.append(1)

    for n in range(2):                                #欠陥画像が少ないのでカサ増し

            #arr2 = img_array1

            arr2 = img_array1[:, ::-1, :]

            #arr2 = random_rotation(arr2, rg=360, row_axis=0, col_axis=1, channel_axis=2)      #ランダムな角度で画像データを回転させて、新しい画像としてカサ増し

            img_array_list.append(arr2)

            cls_list.append(1)

            arr3 = img_array1[::-1, :, :]

            img_array_list.append(arr3)

            cls_list.append(1)

X_train = np.array(img_array_list)

y_train = np.array(cls_list)

print(X_train.shape, y_train.shape)
#テスト画像の読み込み

dir1 = '../input/1056lab-defect-detection-extra/test/'

img_list2 = glob.glob(dir1 + '*.png')

img_list2.sort()

test_list = []



for i in tqdm(img_list2, desc="Loading Test.png files"):

    img2 = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array2 = img_to_array(img2) / 255.0

    test_list.append(img_array2)
test = np.array(test_list)

print(test.shape)
#0と1の要素数を計算

print(list(y_train).count(0))

print(list(y_train).count(1))
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from keras.utils.np_utils import to_categorical



X_learn, X_valid, y_learn, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print('↓学習用データ')

print(X_learn.shape)

print(y_learn.shape)

print('↓検証用データ')

print(X_valid.shape)

print(y_valid.shape)

print('↓学習用教師データのクラス数')

print('正常：', list(y_learn).count(0))

print('欠陥：', list(y_learn).count(1))
import keras.backend as K

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

import tensorflow as tf

from keras import regularizers
from keras.optimizers import SGD, Adam, RMSprop, Nadam

from tensorflow.keras.metrics import AUC

AUC = AUC()



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(units=64, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',

                    loss='binary_crossentropy',

                    metrics=['accuracy', AUC])
model.summary()
from keras.utils.np_utils import to_categorical

Y_learn = to_categorical(y_learn)

Y_valid = to_categorical(y_valid)
#モデルの学習

model.fit(X_learn, Y_learn, epochs=10, batch_size=32)
model.evaluate(X_valid, Y_valid, batch_size=32)
from sklearn.metrics import roc_curve, auc

from sklearn import metrics

predictions = model.predict(X_learn)

a = metrics.roc_auc_score(Y_learn[:,1], predictions[:,1])

print('↓学習データに対する精度')

print(a)
from sklearn import metrics

predictions = model.predict(X_valid)

a = metrics.roc_auc_score(Y_valid[:,1], predictions[:,1])

print('↓未知データに対する精度')

print(a)
#実際の数値と予測値を、データフレームで確認

import pandas as pd

yy = pd.DataFrame(Y_valid[:,1], columns=['y_true'])      #データフレームの作成

aa = pd.DataFrame(predictions[:,1] , columns=['y_pred']) #データフレームの作成

d = pd.concat([yy, aa], axis=1, sort=False)         #列方向に、2つのデータフレームを結合



d['y_pred_No'] = round(d['y_pred'])   #round() → 四捨五入の処理

d.head(50)
Y_train = to_categorical(y_train)

print(list(y_train).count(0))

print(list(y_train).count(1))

print(X_train.shape)

print(Y_train.shape)

print(test.shape)
from keras.optimizers import SGD, Adam, RMSprop, Nadam

from tensorflow.keras.metrics import AUC

AUC = AUC()



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(units=64, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',

                    loss='binary_crossentropy',

                    metrics=['accuracy', AUC])
model.fit(X_train, Y_train, epochs=10, batch_size=32)
pre = model.predict(test)

submit = pd.read_csv('../input/1056lab-defect-detection-extra/sampleSubmission.csv')

submit['defect'] = pre[:,1]

submit.to_csv('pre.csv', index=False)
p = pd.DataFrame(pre[:,1], columns=['pre'])

p.sort_values(by='pre', ascending=False).head(50)