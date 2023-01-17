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
#再現性の確保

#乱数の固定

import os

import numpy as np

import random as rn

import tensorflow as tf



os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(7)

rn.seed(7)
train = pd.read_csv("/kaggle/input/1056lab-student-performance-prediction/train.csv", index_col=0)

test = pd.read_csv("/kaggle/input/1056lab-student-performance-prediction/test.csv", index_col=0)
print(train.shape)

print(test.shape)
#欠損値の有無の確認

print(train.isnull().sum())
print(test.isnull().sum())
train
#ダミー変数の作成

dummy = pd.get_dummies(train[['school', 'class', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']], drop_first=True)
#カテゴリー変数以外を左詰めするため

#ヒートマップ作成のため

train = pd.get_dummies(train, drop_first=True)
train.shape
train
#pandasのオプション

#45列まで表示されるようになる

pd.options.display.max_columns = 45
#ダミー変数以外の変数の相関値

import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 9))

sns.heatmap(train.iloc[:, 0:22].corr(), cmap='BuPu', annot=True,fmt='.2f')
dummy
#相関値の絶対値が0.1以上のもの　＋　作成したダミー変数　を特徴量に。。。

a = train[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'higher', 'internet', 'romantic', 'Dalc', 'Walc']]

#a = train[['age', 'Medu', 'Fedu', 'studytime', 'failures', 'higher']]

#a =  train[['Medu', 'Fedu', 'failures', 'higher']]

X = pd.concat([a, dummy], axis=1, sort=False).values

#X = train.drop('G3', axis=1).values 

y = train['G3'].values

from sklearn.model_selection import train_test_split

X_learn, X_valid, y_learn, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)

print('学習用データ：', X_learn.shape, y_learn.shape)

print('検証用データ', X_valid.shape, y_valid.shape)
X_learn[0]
#kerasの回帰予測用のモデル

import numpy as np

from sklearn import model_selection

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.metrics import r2_score

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, BatchNormalization

from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import SGD, Adam

import tensorflow as tf



def base_model():

    model = Sequential()

    model.add(Dense( units = 30, input_dim = X_learn.shape[1], activation='linear') )

    model.add(BatchNormalization())

    model.add(Dropout(0.5, seed=0))

    model.add( Dense( units = 16, activation='linear') )

    model.add(BatchNormalization())

    model.add(Dropout(0.5, seed=0))

    #model.add(Dense( units = 64, activation='linear') )

    #model.add(BatchNormalization())

    #model.add(Dropout(0.5, seed=0))

    model.add(Dense( units = 1 , activation='linear') )

    model.compile( loss = 'mean_squared_error',  optimizer = 'adam', metrics=['mse'] )

    return model

 

estimator = KerasRegressor( build_fn = base_model,

epochs = 200,

verbose = 2,                

batch_size = 8,

validation_split=0.1

)

 

estimator.fit( X_learn, y_learn, verbose=2)
#学習データに対する評価(RMSE)

from sklearn.metrics import mean_squared_error

import numpy as np

y_true = y_learn

y_pred = estimator.predict(X_learn, batch_size=8)

np.sqrt(mean_squared_error(y_true, y_pred))
#未学習データに対する評価(RMSE)

from sklearn.metrics import mean_squared_error

import numpy as np

y_true = y_valid

y_pred = estimator.predict(X_valid, batch_size=8)

np.sqrt(mean_squared_error(y_true, y_pred))
import numpy as np

from sklearn import model_selection

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.metrics import r2_score

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, BatchNormalization

from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import SGD, Adam

import tensorflow as tf



def base_model():

    model = Sequential()

    model.add(Dense( units = 30, input_dim = X_learn.shape[1], activation='linear') )

    model.add(BatchNormalization())

    model.add(Dropout(0.5, seed=0))

    model.add( Dense( units = 16, activation='linear') )

    model.add(BatchNormalization())

    model.add(Dropout(0.5, seed=0))

    #model.add(Dense( units = 64, activation='linear') )

    #model.add(BatchNormalization())

    #model.add(Dropout(0.5, seed=0))

    model.add(Dense( units = 1 , activation='linear') )

    model.compile( loss = 'mean_squared_error',  optimizer = 'adam', metrics=['mse'] )

    return model

 

estimator = KerasRegressor( build_fn = base_model,

epochs = 200,

verbose = 2,                

batch_size = 8,

validation_split=0.1

)

 

estimator.fit( X, y, verbose=2)
#ダミー変数の作成

dummy1 = pd.get_dummies(test[['school', 'class', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']], drop_first=True)
#a1 = test[['age', 'Medu', 'Fedu', 'studytime', 'failures', 'higher']]

#a1 =  test[['Medu', 'Fedu', 'failures', 'higher']]

a1 = test[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'higher', 'internet', 'romantic', 'Dalc', 'Walc']]
X_test = pd.concat([a1, dummy1], axis=1, sort=False).values

X_test.shape
pre = estimator.predict(X_test, batch_size=8)

pre.shape
pre[0:5]
sample = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv')

sample['G3'] = pre

sample.to_csv('pre.csv', index=False)
sample.shape