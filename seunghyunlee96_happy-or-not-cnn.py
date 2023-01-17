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
# 라이브러리 불러오기

import h5py, cv2

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
train = h5py.File('/kaggle/input/happy-house-dataset/train_happy.h5','r')

test = h5py.File('/kaggle/input/happy-house-dataset/test_happy.h5','r')
train['train_set_x'].shape, train['train_set_y'].shape, test['test_set_x'].shape, test['test_set_y'].shape
fig , axes = plt.subplots(3,3)

fig.set_size_inches(15,15)



for index in range(0,9):

    axes[index//3 , index%3].imshow(train['train_set_x'][index])

    axes[index//3 , index%3].set_title('label : {}'.format(train['train_set_y'][index]))
train['train_set_y'][:5] # 웃으면 1, 웃지 않으면 0 
x_train = np.array(train['train_set_x'][:]) # train셋의 image 데이터

y_train = np.array(train['train_set_y'][:]) # train셋의 label 데이터



x_test = np.array(test['test_set_x'][:])

y_test = np.array(test['test_set_y'][:])
x_train.shape, x_test.shape
y_train.shape, y_test.shape
y_train = y_train.reshape((y_train.shape[0],1))

y_test = y_test.reshape((y_test.shape[0],1))
y_train.shape, y_test.shape
x_train_gray = [] # train 데이터 

for x in x_train:

    img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

    x_train_gray.append(img)
x_test_gray = [] # test 데이터 

for x in x_test:

    img = cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)

    x_test_gray.append(img)
fig , axes = plt.subplots(2,2)

fig.set_size_inches(10,10)



for index in range(0,4):

    axes[index//2 , index%2].imshow(x_train_gray[index], cmap = 'gray')

    axes[index//2 , index%2].set_title('label : {}'.format(y_train[index]))
x_train_gray = np.array(x_train_gray)

x_test_gray = np.array(x_test_gray)
# x_train = x_train_gray.reshape((-1,64,64,1)) # 흑백 전처리

# x_test = x_test_gray.reshape((-1,64,64,1))



x_train.shape, y_train.shape, x_test.shape, y_test.shape
train_datagen = ImageDataGenerator(

    rescale = 1/255.0,

    brightness_range = [0.5, 1.5], # 밝기 0.5~1.5 랜덤 설정

    zoom_range = [0.8,1.1],        # 0.8~1.1 사이로 임의로 크기 설정

    rotation_range =15.0,           # 회전 범위 

    channel_shift_range = 25,      # 부동소수점. 무작위 채널 이동의 범위.

    horizontal_flip = True)        # 좌우반전



test_datagen = ImageDataGenerator(

    rescale = 1/255.0)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
batch_size = 16 # 배치 사이즈 지정



train_gen = train_datagen.flow(x_train, y_train, batch_size = batch_size, shuffle = True)

test_gen = test_datagen.flow(x_test, y_test, batch_size = batch_size, shuffle = False)
model = Sequential([

    Conv2D(64, (3,3), activation = 'relu', input_shape = (64,64,3)), 

    #MaxPooling2D(),

    

#     Conv2D(32, (3,3), activation = 'relu'),

#     MaxPooling2D(),

    

#     Conv2D(16, (3,3), activation = 'relu'),

#     MaxPooling2D(),

    

    Flatten(),

    Dense(32, activation = 'relu'),

    Dense(1, activation = 'sigmoid')

])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc']) # 최적화함수는 adam을 사용해줍니다.



checkpath = './ck.ckpt'

checkpoint = ModelCheckpoint(

    filepath = checkpath,

    save_best_only = True,

    save_weights_only = True, # 가중치만 저장

    verbose =1,

    monitor = 'val_acc' # val_acc를 모니터하며 가장 높은 값을 저장

    )
model.fit(train_gen, 

          validation_data = test_gen,

          epochs = 20,

          callbacks = [checkpoint])
model.load_weights(checkpath)
x_test_input = x_test.copy().astype(np.float64)

x_test_input -= np.mean(x_test, keepdims=True)

x_test_input /= (np.std(x_test, keepdims=True) + 1e-6)
x_test = x_test / 255.0 # 정규화
y_pred = model.predict(x_test) # 예측
y_pred_logical = (y_pred > 0.5).astype(np.int) # 0.5를 임계점으로 잡겠습니다.
print('test acc: %s'% accuracy_score(y_test, y_pred_logical))
cm = confusion_matrix(y_test, y_pred_logical)

sns.heatmap(cm, annot = True)