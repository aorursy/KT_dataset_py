# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'../input/train.csv',index_col='Id')

test = pd.read_csv(r'../input/test.csv',index_col='Id')
sub = pd.read_csv(r'../input/sample_submission.csv')
import pandas as pd

import numpy as np

from sklearn import datasets





import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import math



from sklearn.metrics import r2_score



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

import os

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split 

import pickle

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

p = pd.DataFrame()

for i in train.columns:

    if train[i].dtype == 'object':

        train[i] = train[i].astype('category')

        if train[i].isna().sum() != 0:

            p[i] = train[i].fillna(train[i].value_counts().keys()[0])

#             p[i+'_cat'] = train[i].cat.codes

        else:

            p[i] = train[i]

    else:

        p[i] = train[i].fillna(method = 'ffill')

        

T = pd.DataFrame()

for j in test.columns:

    if test[j].dtype == 'object':

        test[j] = test[j].astype('category')

        if test[j].isna().sum() != 0:

            T[j] = test[j].fillna(test[j].value_counts().keys()[0])

#             T[j+'_cat'] = test[j].cat.codes

        else:

            T[j] = test[j]

    else:

        T[j] = test[j].fillna(method = 'ffill')
plt.figure(figsize=(15,7))

sns.boxplot(p['YrSold'],p['SalePrice'],data=p)

# sns.boxplot(T['YrSold'],T['PoolArea'],data=T)

plt.show()
plt.figure(figsize=(15,15))

sns.heatmap(p.corr(),square=True)

plt.show()
p = pd.get_dummies(p)

T = pd.get_dummies(T)

p = pd.concat([p[T.columns.values],p[['SalePrice']]],axis=1)
p.shape
import tensorflow as tf

import tflearn
np.random.seed(1337)

tf.reset_default_graph()

r2 = tflearn.R2()

net = tflearn.input_data(shape=[None,270])

net = tflearn.fully_connected(net,370,activation='linear')

net = tflearn.fully_connected(net,150, activation='linear')

net = tflearn.fully_connected(net,50, activation='linear')

net = tflearn.fully_connected(net,1, activation='linear')

sgd = tflearn.SGD(learning_rate=0.1,lr_decay=0.01,decay_step=100)

net = tflearn.regression(net,optimizer=sgd,loss='mean_square',metric=r2)

model = tflearn.DNN(net)



model.fit(p.drop(['SalePrice'],axis=1).values,np.reshape(p['SalePrice'].values,(-1,1)),show_metric=True,shuffle=True,n_epoch=1000,validation_set=0.2)



pred = model.predict(p.drop(['SalePrice'],axis=1).values)

plt.figure(figsize=(15,5))

plt.plot(pred[:100],label='predicted')

plt.plot(p['SalePrice'][:100],label='Actual')

plt.legend()

plt.show()
# from sklearn.model_selection import GridSearchCV

# params = {'n_estimators':[50,100,200,300,400],

#          'learning_rate':[0.01,0.1,0.5,1],

#          'max_depth' : [1,2,3,4,5]}

# model = XGBRegressor()

# clf = GridSearchCV(model,params,cv=5)

# clf.fit()

# clf.best_params_ 
# model = XGBRegressor(params=clf.best_params_)

# model.fit(p.drop(['SalePrice'],axis=1).values,p['SalePrice'].values)

res = model.predict(T.values)

sub['SalePrice'] = res

sub.to_csv('submisson.csv',index=False)
# np.random.seed(1337)

# import keras

# from keras import metrics

# from keras import regularizers

# from keras.models import Sequential

# from keras.layers import Dense, Dropout, Flatten, Activation

# from keras.layers import Conv2D, MaxPooling2D

# from keras.optimizers import Adam, RMSprop

# from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# from keras.utils import plot_model

# from keras.models import load_model
# p.shape
# np.random.seed(1337)

# model = Sequential()

# model.add(Dense(270, input_dim=270, kernel_initializer='normal', activation='relu'))

# model.add(Dense(100, kernel_initializer='normal', activation='relu'))

# model.add(Dense(60, kernel_initializer='normal', activation='relu'))

# model.add(Dense(30, kernel_initializer='normal', activation='relu'))

# model.add(Dense(6, kernel_initializer='normal', activation='relu'))



# model.add(Dense(1, kernel_initializer='normal'))

# # Compile model

# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(p.drop(['SalePrice'],axis=1).values,p['SalePrice'].values,batch_size=10,epochs=1000)

# res = model.predict(T.values)

# sub['SalePrice'] = res

# sub.to_csv('submisson.csv',index=False)

# sub
# sub