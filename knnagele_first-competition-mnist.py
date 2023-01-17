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
# 加载数据

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv',delimiter=',')

print(train.head())

print(train.columns)
# 数据分布研究

print(train.describe())

print((train.shape[0] - train.count())/(train.shape[0]))
# 将df数据转化为tensor_slices

# def df_to_dataset(df):

#     df = df.copy()

#     labels = df.pop('label')

#     ds = tf.data.Dataset.from_tensor_slices((dict(df),labels))

#     return ds



# train_ds = df_to_dataset(train)

# print(train_ds.take(1))

data = train.copy()

train_y = data['label'].values

data.pop('label')

train_x = data.values



# 数据归一化

train_x = train_x / 255

print(train_x.shape, ' ', train_y.shape)

print(train_x)
# 构建模型

import tensorflow as tf

from tensorflow.keras import layers

import matplotlib.pyplot as plt



model = tf.keras.Sequential(

    [

        

        layers.Dense(64,activation='relu',kernel_initializer='he_normal', input_shape=(784,)),

        layers.Dense(64,activation='relu',kernel_initializer='he_normal'),

        layers.Dense(64,activation='relu',kernel_initializer='he_normal'),

        layers.Dense(64,activation='relu',kernel_initializer='he_normal'),

        layers.Dense(10,activation='softmax')

    ]

)



model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

model.summary()
# 模型训练

history = model.fit(train_x, train_y, batch_size=256, epochs=100, validation_split=0.3, verbose=1)

print(history.history['accuracy'],history.history['val_accuracy'])



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['training', 'validation'], loc='upper left')



plt.show()
# 模型评估

result = model.evaluate(train_x,train_y)

print(result)
# 预测数据

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_id = test.id



test = test.drop('id',axis=1)

test = test/255



y_pre = model.predict(test)

print(y_pre)

y_pre = np.argmax(y_pre,axis=1)

print(y_pre)
# 保存数据

sample_sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

print(sample_sub)



sample_sub['id'] = test_id

sample_sub['label'] = y_pre

sample_sub.to_csv('/kaggle/working/submission.csv',index=False)



print(sample_sub.head())