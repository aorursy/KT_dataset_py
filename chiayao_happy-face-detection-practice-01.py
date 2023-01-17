'''

下面是Kaggle預先幫你設定好的程式碼，

執行之後會看到

/kaggle/input/happy-house-dataset/test_happy.h5

/kaggle/input/happy-house-dataset/train_happy.h5

所提供的要進行輸入的資料集檔案都會放在 ../input/ 裡面

這邊已經打包好成為 h5py 的格式了

'''



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
'''

來看看Kaggle給我哪些硬體？

'''

# !nvidia-smi --help

print("以下是GPU的部分")

!nvidia-smi -L



#check CPU

print("n以下是CPU的部分")

!lscpu
'''

接下來，用 h5py 讀入檔案

'''

import h5py

train_file = h5py.File('/kaggle/input/happy-house-dataset/train_happy.h5')

test__file = h5py.File('/kaggle/input/happy-house-dataset/test_happy.h5')

# 這種 h5 檔案有點類似 python 的字典檔

# 我們可以看看這裡面有哪些 keys

print('在訓練檔的keys有', list(train_file.keys()))

print('在測試檔的keys有', list(test__file.keys()))

# 個別讀出這些 HDF5 的 dataset

train_h5_x = train_file['train_set_x']

train_h5_y = train_file['train_set_y']

test__h5_x = test__file['test_set_x']

test__h5_y = test__file['test_set_y']

print('現在train_h5_x的形狀是：', train_h5_x.shape)

number, w, h, channel = train_h5_x.shape

print(f'這代表訓練集有{number}張，寬{w}與長{h}，的 {channel} channel 的圖片')

print('現在test__h5_x的形狀是：', test__h5_x.shape)

number, w, h, channel = test__h5_x.shape

print(f'這代表測試集有{number}張，寬{w}與長{h}，的 {channel} channel 的圖片')

print('現在train_h5_y的形狀是：', train_h5_y.shape)

print(f'這代表測試集的{train_h5_y.shape[0]}個答案')

print('現在test__h5_y的形狀是：', test__h5_y.shape)

print(f'這代表訓練集的{test__h5_y.shape[0]}個答案')
'''

這邊來試著看看圖片實際上的樣子

'''

import matplotlib.pyplot as plt

plt.imshow(train_h5_x[30])

print(f'train_h5_y[30] 的數值為 {train_h5_y[30]}')
'''

接下來，必須把上面提取出來的 h5py 物件，

轉換為 NumPy 的 array

才方便送到 model 裡做訓練

'''

import numpy as np

train_x = np.array(train_h5_x[:])

test__x = np.array(test__h5_x[:])

train_y = np.array(train_h5_y[:])

test__y = np.array(test__h5_y[:])

print(f'train_x的形狀為{train_x.shape}')

print(f'test__x的形狀為{test__x.shape}')

print(f'train_y的形狀為{train_y.shape}')

print(f'test__y的形狀為{test__y.shape}')
'''

資料的前處理

'''

# 這邊要把原本數值介於0到255的圖片，轉換為介於-1到1之間，達到標準化

train_x = (train_x/127.5)-1

test__x = (test__x/127.5)-1

# 接著，要把答案變成一一對應的 array

print(f'原本train_y的形狀{train_y.shape}')

print(f'原本test__y的形狀{test__y.shape}')

train_y = train_y.reshape((1, train_y.shape[0]))

test__y = test__y.reshape((1, test__y.shape[0]))

print(f'reshape後train_y的形狀{train_y.shape}')

print(f'reshape後test__y的形狀{test__y.shape}')

train_y = train_y.T

test__y = test__y.T

print(f'再經轉置後train_y的形狀{train_y.shape}')

print(f'再經轉置後test__y的形狀{test__y.shape}')
'''

接下來，來建置模型的架構吧

'''

# Keras 的起手式

from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D



model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='Same', input_shape=(64,64,3)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='Same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='Same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# 按照往例，查看一下模型摘要

print(model.summary())

# 定義模型的訓練方式 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

準備訓練模型

'''

epoch_total = 200

epoch_now = 1

batch_size = 50



import csv

!touch /kaggle/input/result.csv

result_path = '/kaggle/input/result.csv'

with open(result_path, 'w', newline='') as result_csv:

    writer = csv.writer(result_csv)

    writer.writerow(['n_epoch','train_acc','test_acc'])

while epoch_now <= epoch_total:

    train_hx = model.fit(x=train_x, y=train_y, epochs=1, verbose=2, batch_size=batch_size)

    test_result = model.evaluate(test__x, test__y, verbose=1)

    train_acc = train_hx.history['acc'][-1] * 100

    test__acc = test_result[1] * 100

    with open(result_path, 'a', newline='') as result_csv:

        writer = csv.writer(result_csv)

        writer.writerow([epoch_now,train_acc,test__acc])

    print(f'這是第{epoch_now}個epoch，而訓練集的準確率是{train_acc}%，測試集的準確率是{test__acc}%')

    epoch_now += 1

print('訓練結束！')
'''

把結果畫出來

'''

import pandas as pd

result_df = pd.read_csv(result_path)

result_df.plot.line(

    x='n_epoch',

    y=['train_acc','test_acc'],

    figsize = (10,5))