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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
boston = pd.read_csv('../input/boston-housing-dataset/HousingData.csv')
boston
boston.keys()
boston.info()
boston.isnull().sum()
boston[boston.CRIM.isnull()]
boston[boston.ZN.isnull()]
boston.describe()
import matplotlib.pyplot as plt
import numpy as np

boston_CRIM = boston.CRIM
boston_RM = boston.RM

plt.scatter(boston_RM, boston_CRIM, marker = 'o')
plt.scatter(boston_RM.mean(), boston_CRIM.mean(), marker = '^')
plt.scatter(np.median(boston_RM), np.median(boston_CRIM), marker = 'v')

plt.title('Boston Crime-Rooms comparison Plot')
plt.ylabel('Crime Rate')
plt.xlabel('No. of Rooms')
plt.legend(['CR_relation','mean','median'])
plt.show()
boston_DIS = boston.DIS
boston_PTR = boston.PTRATIO

plt.scatter(boston_DIS, boston_PTR, marker = 'o', color = 'orange')
plt.scatter(boston_DIS.mean(), boston_PTR.mean(), marker = '^', color = 'black')
plt.scatter(np.median(boston_DIS), np.median(boston_PTR), marker = 'v', color = 'blue')

plt.title('Boston Distances to Centers - PTRATIO Comparison Plot')
plt.ylabel('Pupil-Teacher Ratio')
plt.xlabel('Distances to Centeres')
plt.legend(['DPT_relation', 'mean', 'median'])
plt.show()
X = boston.drop(columns = 'MEDV')
Y = boston.MEDV
X
# you could not make boxplot with dataframe. so, make it as numpy array.

X_np = X.to_numpy()
plt.boxplot(X_np)
plt.show()
X.describe()
X_mean = X.fillna(X.mean())
X_mean.describe()
X_mean = X_mean.to_numpy()
plt.boxplot(X_mean)
plt.show()
boston.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaler = scaler.fit_transform(X_mean)

plt.boxplot(X_scaler)
plt.show()
X_scaler.shape
X_pd = pd.DataFrame(X_scaler, columns = X.columns)
X_pd
plt.boxplot(X_scaler)
plt.show()
X_pd[X_pd.CRIM > np.percentile(X_pd.CRIM,25)] = np.percentile(X_pd.CRIM,25)
X_pd[X_pd.CRIM < np.percentile(X_pd.CRIM,75)] = np.percentile(X_pd.CRIM,75)
X_pd2 = X_pd.to_numpy

plt.boxplot(X_pd2)
plt.show()
# np.percentile(X_pd.CRIM, 25), np.percentile(X_pd.CRIM, 75)
X_pd
X = boston.drop(columns=['MEDV'])
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(X,Y,test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,test_size = 0.2)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)
lr.score(x_test, y_test)
from sklearn import metrics

pred_lr = lr.predict(x_test)
metrics.r2_score(y_test, pred_lr)
from sklearn.linear_model import ElasticNet
en_9 = ElasticNet(alpha=0.9)
en_9.fit(x_train, y_train)
en_9.score(x_test, y_test)

import tensorflow as tf
from tensorflow.keras import layers

regular = 0.1 # regularization amount

metrics_nm = ['accuracy','mean_squared_error']

model = tf.keras.Sequential()
model.add(layers.Input(shape=x_train.shape[1]))
model.add(layers.Dense(32, activation='relu',
         kernel_regularizer = tf.keras.regularizers.l2(regular),  # Dense Regularization
         activity_regularizer = tf.keras.regularizers.l2(regular)))  # Dense Regularization
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='SGD', loss='mse', metrics=metrics_nm)

hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val,y_val))
import matplotlib.pyplot as plt

weights, biases = model.layers[1].get_weights()
print(weights.shape, biases.shape)

plt.subplot(212)
plt.plot(weights,'x')
plt.plot(biases, 'o')
plt.title('L2 - 0.1')

plt.subplot(221)
plt.plot(hist.history['accuracy'],'^--',label='accuracy')
plt.plot(hist.history['val_accuracy'],'^--', label='v_accuracy')
plt.legend()
plt.title('L2 - 0.1')

plt.subplot(222)
plt.plot(hist.history['loss'],'x--',label='loss')
plt.plot(hist.history['val_loss'],'x--', label='val_loss')
plt.legend()
plt.title('L2 - 0.1')

plt.show()
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.3)
x_test = standardScaler.transform(x_test)
pred = model.predict(x_test)
model.evaluate(x_test, y_test, batch_size=16)
from sklearn import metrics

metrics.r2_score(y_test, pred)
