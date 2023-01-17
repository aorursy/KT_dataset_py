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
import pandas as pd

import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()
x_train, x_test, y_train, y_test= train_test_split(cancer.data, cancer.target, stratify=cancer.target)
#Normalizer(columns가 아니라 row마다 각각 정규화 so 유클리드 거리=1)로 데이터 스케일링(fit:데이터 변환 학습, transform:실제 스케일 조정)

from sklearn.preprocessing import Normalizer
scaler=Normalizer()

x_train_scale=scaler.fit_transform(x_train)

x_test_scale=scaler.transform(x_test)
#scaling 이후 학습성능 향상 확인

from sklearn.svm import SVC     #SVM은 인공신경망의 특수한 형태로 layer를 많이 쌓지 않아도 우수한 성능을 보여준다. 

svc=SVC()

svc.fit(x_train, y_train)

print('test accuracy :%.3f' %(svc.score(x_test, y_test)))
svc.fit(x_train_scale, y_train)

print('after scaling test accuracy :%.3f' %(svc.score(x_test_scale, y_test)))
#tensorflow keras 적용

import tensorflow as tf

from tensorflow.keras import layers

import numpy as np

from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
model=tf.keras.Sequential()

model.add(layers.Input(shape=x_train.shape[1]))
model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(1, activation='relu'))  
metrics_nm = ['accuracy','mean_squared_error','binary_accuracy','binary_crossentropy']

model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=metrics_nm)

hist=model.fit(x_train_scale, y_train, epochs=200, batch_size=20, validation_split=0.2, verbose=0)  

#fit하면서 validation_split을 하는 이유? => validation_set과 train_set의 history.keys 값을 확인해 일치하는지 본다. 

#sample size와 batch size정하기?

#hist는 print하는 의미가 없는 듯?

hist.history.keys()  #loss와 평가방법(metrics)인, 예를 들어 accuracy, MSE의 epoch별 변화를 보여주는 것. train set과 validation set 오차가 적은지 확인하고 비로소 test set으로 evaluate 함

plt.plot(hist.history['binary_crossentropy'])

plt.plot(hist.history['val_binary_crossentropy'])
plt.plot(hist.history['binary_accuracy'])

plt.plot(hist.history['val_binary_accuracy'])
model.evaluate(x_test_scale, y_test, batch_size=20)
y_pred=model.predict(x_test_scale)
#Classification report

from sklearn.metrics import classification_report

y_pred_class=np.where(y_pred>0.5, 1, 0)

print(classification_report(y_test, y_pred_class))