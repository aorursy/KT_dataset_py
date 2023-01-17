import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score,roc_curve, auc

import numpy as np

import seaborn as sns

from sklearn import preprocessing, metrics

import lightgbm as lgb

import gc

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

%matplotlib inline

gc.enable()

import matplotlib.pyplot as plt

import datetime

from sklearn import preprocessing

import re

from sklearn.model_selection import StratifiedKFold, KFold

print('Libraries Imported')
import os 

os.listdir('../input/iit_traffic_data/iit_traffic_data/')

path = '../input/iit_traffic_data/iit_traffic_data/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
train.head()
test.head()
ax = sns.countplot(x='is_holiday', data=train)

ax.set_xlabel('is_holiday')

ax.set_ylabel("count")  

ax = sns.countplot(x='weather_type', data=train)

ax.set_xlabel('weather_type')

ax.set_ylabel("count")  

ax = sns.countplot(x='weather_type', data=train)

ax.set_xlabel('weather_type')

ax.set_ylabel("count")  





train.describe()
# count the number of NaN values in each column

print(train.isnull().sum())
train.shape

len(train.columns)
i = 0

while i < len(train.columns):    



    train[train.columns[i:i+10]].hist()



    i += 10
i = 0

while i < len(train.columns):    

    plt.figure(figsize=(10,10))

    ax = sns.boxplot(data=train[train.columns[i:i+10]],  palette="Set2")

    plt.title('Box Plot')

    plt.show()

    i += 10
plt.figure(figsize=(10,10))

sns.heatmap(train.corr(), cmap='RdBu_r', annot=True, center=0.0)

plt.title('Correlation between columns')

plt.show()
def run_lgbm(train_X, train_y):

    params = {  

        'num_leaves':30,

            'objective':'regression',

            "metric" : "rmse",

            'max_depth':7,

            'learning_rate':.001,

            'max_bin':100,

            "bagging_fraction" : 0.7,

            "feature_fraction" : 0.5,

            "bagging_frequency" : 6,

            "bagging_seed" : 42,

            "verbosity" : 1,

#           "boosting": ['rf'],

           "reg_alpha" : 0,

           'reg_lambda' :10

             }



    lgbm_train = lgb.Dataset(train_X, label=train_y)





    model = lgb.train(params, lgbm_train,num_boost_round=150)



    print(datetime.datetime.now())

    return model
train.head()
cat_cols = ['is_holiday','weather_type']

ohe = pd.get_dummies(train[cat_cols],  drop_first=False)

ohe.columns

train.drop(cat_cols, axis=1,inplace=True)

train= pd.concat([train, ohe], axis=1)

print(train.head())
features = [c for c in train.columns if c not in ['date_time', 'traffic_volume','weather_description',

                                                 ]]

target = ['traffic_volume']

X = train[features].values

y = train[target]
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.20, random_state=1234)

print("Training, validation split  done")
#predicting on test set

model = run_lgbm(train_X, train_y)





val_y_pred=model.predict(val_X)





fig, ax = plt.subplots(figsize=(12,10))

lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()



from sklearn import metrics

from sklearn.metrics import r2_score



# print('Intercept: \n', regressor.intercept_)

# print('Coefficients: \n', regressor.coef_)



print('R Square value', r2_score(val_y,val_y_pred))

print('mean absolute error', metrics.mean_absolute_error(val_y,val_y_pred))

print('root mean_squared_error', np.sqrt(metrics.mean_squared_error(val_y,val_y_pred)))
from keras.layers.core import Activation



# from resnets_utils import *



from keras.models import load_model



from keras.utils import np_utils



from keras import applications



from keras import optimizers

from keras.models import Sequential,Model,load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
NN_model = Sequential()



# The Input Layer :

NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))



# The Hidden Layers :

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))



# The Output Layer :

NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))



# Compile the network :

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.summary()





NN_model.fit(train_X, train_y, epochs=50, batch_size=32)



pred_val_y = NN_model.predict(val_X)

print('R Square value', r2_score(val_y,pred_val_y))

print('mean absolute error', metrics.mean_absolute_error(val_y,pred_val_y))

print('root mean_squared_error', np.sqrt(metrics.mean_squared_error(val_y,pred_val_y)))