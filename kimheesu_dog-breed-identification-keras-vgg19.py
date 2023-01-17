# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.applications.vgg19 import VGG19

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten

import matplotlib.pyplot as plt #그림을 그려주는 라이브러리





import os

from tqdm import tqdm #돌고있는 내용을 보여주는

from sklearn import preprocessing #데이터분석 라이브러리

from sklearn.model_selection import train_test_split

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
%matplotlib inline
!ls ../input
df_train = pd.read_csv('../input/labels.csv')

df_test = pd.read_csv('../input/sample_submission.csv')
df_train.head(5)
df_test.head(5)

# 아래를 보면 one hot 형식으로 되어있음을 알 수 있다.
targets_series = pd.Series(df_train['breed']) 

#breed와 index를 Series구조화



one_hot = pd.get_dummies(targets_series, sparse = True) 

# spare data(밀도가 낮은 데이터)? - https://blog.naver.com/qbxlvnf11/221429203293

# get_dummies는 위와 같이 one hot 형식으로 나타내게 해주는 함수(one hot encoding) - https://homeproject.tistory.com/4
one_hot_labels = np.asarray(one_hot)

#one hot형식으로 인코딩 한 내용을 다시 어레이 형태로 변환
im_size = 128

#224로 하면 RAM부족 ㅠㅠ



x_train = []

y_train = []

x_test = []
i = 0 

for f, breed in tqdm(df_train.values): #tqdm은 반복문의 진행상황을 알려준다.

    img = cv2.imread('../input/train/{}.jpg'.format(f))

    label = one_hot_labels[i]

    x_train.append(cv2.resize(img, (im_size, im_size)))

    y_train.append(label)

    i += 1###############################????
for f in tqdm(df_test['id'].values):

    img = cv2.imread('../input/test/{}.jpg'.format(f))

    x_test.append(cv2.resize(img, (im_size, im_size)))
y_train_raw = np.array(y_train, np.uint8)

#리스트 형태를 넘파이 ndarray형태로

x_train_raw = np.array(x_train, np.float32) / 255.

x_test  = np.array(x_test, np.float32) / 255. 

##255로 나누는 것은 RGB값
print(x_train_raw.shape)

print(y_train_raw.shape)

print(x_test.shape)
num_class = 120

# 120 breeds
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)
# Create the base pre-trained model

# Can't download weights in the kernel

base_model = VGG19(#weights='imagenet',

    weights = None, 

    include_top=False, input_shape=(im_size, im_size, 3))

##Dense layer(fully connected layers)는 모두 연결되어 있어 include_top=False를 하고 input_shape를 변경해야한다.

##참고링크(https://rarena.tistory.com/entry/keras-%ED%8A%B9%EC%A0%95-%EB%AA%A8%EB%8D%B8%EB%A1%9C%EB%93%9C%ED%95%98%EC%97%AC-%EB%82%B4-%EB%A0%88%EC%9D%B4%EC%96%B4)



# 내가 붙일 레이어 2개를 기존의 VGG19모델에 붙여 넣는다.



x = base_model.output

x = Flatten()(x)

predictions = Dense(num_class, activation='softmax')(x)



# 우리가 트레인 시킬 모델 완성!

model = Model(inputs=base_model.input, outputs=predictions)



# First: train only the top layers (which were randomly initialized)

for layer in base_model.layers:

    layer.trainable = False



model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])



callbacks_list = [ #keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1),

                 keras.callbacks.ModelCheckpoint('my_model.h5', monitor='val_loss', verbose=1, save_best_only=True)]

model.summary()



##MaxPooling에 관한 설명 (https://blog.naver.com/pgh7092/221106015450)
hist = model.fit(X_train, Y_train, epochs=50, validation_data=(X_valid, Y_valid), verbose=1, callbacks=callbacks_list)
#  모델 학습 과정 표시하기

%matplotlib inline

import matplotlib.pyplot as plt



fig, loss_ax = plt.subplots()



acc_ax = loss_ax.twinx()



loss_ax.plot(hist.history['loss'], 'y', label='train loss')

loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')



acc_ax.plot(hist.history['acc'], 'b', label='train acc')

acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')



loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('loss')

acc_ax.set_ylabel('accuray')



loss_ax.legend(loc='upper left')

acc_ax.legend(loc='lower left')



plt.show()
preds = model.predict(x_test, verbose=1)