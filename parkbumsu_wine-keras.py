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
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
wine= datasets.load_wine()
wine.data
wine.target
X = wine.data
y = wine.target
# tartget의 종류가 3가지이므로 더미화
y = pd.get_dummies(y)
#데이터 값 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 )
#딥러닝 신경망 구축.
model = tf.keras.Sequential() # model
model.add(layers.Input(shape=x_train.shape[1])) # 층 생성후 연결
model.add(layers.Dense(169, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics='accuracy') # 학습 방법
hist = model.fit(x_train,y_train, epochs=100, batch_size=100, validation_split=0.2,verbose=1) 
model.evaluate(x_test,y_test, batch_size=128) # 평가
hist.history.keys()
# 결과 값 시각화
plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['accuracy'],label='acc')
plt.plot(hist.history['val_loss'],label='v_loss')
plt.plot(hist.history['val_accuracy'],label='v_acc')
plt.legend()
