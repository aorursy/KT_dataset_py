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
from sklearn.decomposition import KernelPCA

from sklearn.preprocessing import StandardScaler

from sklearn.multiclass import OneVsRestClassifier

from sklearn.multioutput import MultiOutputRegressor

from sklearn.utils import class_weight

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

from sklearn.utils import class_weight

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from skmultilearn.model_selection import iterative_train_test_split

import pickle

from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input

from keras.models import Sequential, Model

from keras import optimizers, regularizers, initializers

from keras.callbacks import ModelCheckpoint, Callback

from keras import backend as K

from keras.optimizers import Adam

import tensorflow as tf

from xgboost import XGBRegressor

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
NCA1 = 100

NCA2 = 100

DROPRATE = 0.2

EP = 500

BATCH_SIZE = 256

VAL_RATIO = 0.1

TEST_RATIO = 0.1
lipophilicity_df= pd.read_csv('../input/lipophilicity_descriptors/Lipophilicity_df_revised.csv')

print(lipophilicity_df.shape)

lipophilicity_df.head()
lipophilicity_df.drop(['smiles','CMPD_CHEMBLID'], axis=1, inplace=True)

print(lipophilicity_df.shape)

lipophilicity_df.head()
# Get indices of NaN

#inds = pd.isnull(tox21_df).any(1).nonzero()[0]
# Drop NaN from the dataframe

#tox21_df.dropna(inplace=True)

#print(tox21_df.shape)

#tox21_df.head()
lipophilicity_df = lipophilicity_df.fillna(0)

lipophilicity_df.head()
lipophilicity_descriptors_df= pd.read_csv('../input/lipophilicity_descriptors/Lipophilicity_descriptors_df.csv',low_memory=False)

print(lipophilicity_descriptors_df.shape)

lipophilicity_descriptors_df.head()
# function to coerce all data types to numeric



def coerce_to_numeric(df, column_list):

    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
coerce_to_numeric(lipophilicity_descriptors_df, lipophilicity_descriptors_df.columns)

lipophilicity_descriptors_df.head()
lipophilicity_descriptors_df = lipophilicity_descriptors_df.fillna(0)

lipophilicity_descriptors_df.head()
#tox21_descriptors_df.drop(tox21_descriptors_df.index[inds],inplace=True)

#tox21_descriptors_df.shape
lipophilicity_scaler1 = StandardScaler()

lipophilicity_scaler1.fit(lipophilicity_descriptors_df.values)

lipophilicity_descriptors_df = pd.DataFrame(lipophilicity_scaler1.transform(lipophilicity_descriptors_df.values),

                                            columns=lipophilicity_descriptors_df.columns)
nca = NCA1

cn = ['col'+str(x) for x in range(nca)]
lipophilicity_transformer1 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)

lipophilicity_transformer1.fit(lipophilicity_descriptors_df.values)

lipophilicity_descriptors_df = pd.DataFrame(lipophilicity_transformer1.transform(lipophilicity_descriptors_df.values),

                                            columns=cn)

print(lipophilicity_descriptors_df.shape)

lipophilicity_descriptors_df.head()
X_train, X_test, y_train, y_test = train_test_split(lipophilicity_descriptors_df.values,

                                                    lipophilicity_df.values, random_state=32,

                                                    test_size=TEST_RATIO)
parameters = {'estimator__kernel':['rbf'], 

              'estimator__epsilon':[0.1,0.25,0.5],

              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}

lipophilicity_svr = GridSearchCV(MultiOutputRegressor(SVR()), 

                        parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
result = lipophilicity_svr.fit(X_train, y_train)
pred = lipophilicity_svr.predict(X_test)

svr_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(svr_rmse)
def huber_loss(y_true, y_pred):

        return tf.losses.huber_loss(y_true,y_pred)
lipophilicity_model = Sequential()

lipophilicity_model.add(Dense(128, input_dim=lipophilicity_descriptors_df.shape[1], 

                              kernel_initializer='he_uniform'))

lipophilicity_model.add(BatchNormalization())

lipophilicity_model.add(Activation('tanh'))

lipophilicity_model.add(Dropout(rate=DROPRATE))

lipophilicity_model.add(Dense(64,kernel_initializer='he_uniform'))

lipophilicity_model.add(BatchNormalization())

lipophilicity_model.add(Activation('tanh'))

lipophilicity_model.add(Dropout(rate=DROPRATE))

lipophilicity_model.add(Dense(32,kernel_initializer='he_uniform'))

lipophilicity_model.add(BatchNormalization())

lipophilicity_model.add(Activation('tanh'))

lipophilicity_model.add(Dropout(rate=DROPRATE))

lipophilicity_model.add(Dense(lipophilicity_df.shape[1],kernel_initializer='he_uniform',activation=None))
lipophilicity_model.compile(loss=huber_loss, optimizer='adam',metrics=['mse'])
checkpoint = ModelCheckpoint('lipophilicity_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
hist = lipophilicity_model.fit(X_train, y_train, 

                               validation_split=VAL_RATIO,epochs=EP, batch_size=BATCH_SIZE, 

                               callbacks=[checkpoint])
plt.ylim(0., 1.)

plt.plot(hist.epoch, hist.history["loss"], label="Train loss")

plt.plot(hist.epoch, hist.history["val_loss"], label="Valid loss")
lipophilicity_model.load_weights('lipophilicity_model.h5')
pred = lipophilicity_model.predict(X_test)

nn_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(nn_rmse)
inp = lipophilicity_model.input

out = lipophilicity_model.layers[-2].output

lipophilicity_model_gb = Model(inp, out)
X_train = lipophilicity_model_gb.predict(X_train)

X_test = lipophilicity_model_gb.predict(X_test)
data = np.concatenate((X_train,X_test),axis=0)
lipophilicity_scaler2 = StandardScaler()

lipophilicity_scaler2.fit(data)

X_train = lipophilicity_scaler2.transform(X_train)

X_test = lipophilicity_scaler2.transform(X_test)
data = np.concatenate((X_train,X_test),axis=0)
nca = NCA2
lipophilicity_transformer2 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)

lipophilicity_transformer2.fit(data)

X_train = lipophilicity_transformer2.transform(X_train)

X_test = lipophilicity_transformer2.transform(X_test)
nca = X_train.shape[1]

parameters = {'estimator__kernel':['rbf'], 

              'estimator__epsilon':[0.1,0.25,0.5],

              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}

lipophilicity_svr_gb = GridSearchCV(MultiOutputRegressor(SVR()), 

                                    parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
result = lipophilicity_svr_gb.fit(X_train, y_train)
pred = lipophilicity_svr_gb.predict(X_test)

svr_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(svr_gb_rmse)
parameters = {'estimator__learning_rate':[0.05,0.1,0.15],'estimator__n_estimators':[75,100,125], 'estimator__max_depth':[3,5,7],

              'estimator__booster':['gbtree','dart'],'estimator__reg_alpha':[0.1,0.05],'estimator__reg_lambda':[0.5,1.]}



lipophilicity_xgb_gb = GridSearchCV(MultiOutputRegressor(XGBRegressor(random_state=32)), 

                                    parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
result = lipophilicity_xgb_gb.fit(X_train, y_train)
pred = lipophilicity_xgb_gb.predict(X_test)

xgb_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(xgb_gb_rmse)
with open('lipophilicity_svr.pkl', 'wb') as fid:

    pickle.dump(lipophilicity_svr, fid)

with open('lipophilicity_transformer1.pkl', 'wb') as fid:

    pickle.dump(lipophilicity_transformer1, fid)

with open('lipophilicity_transformer2.pkl', 'wb') as fid:

    pickle.dump(lipophilicity_transformer2, fid)

with open('lipophilicity_scaler1.pkl', 'wb') as fid:

    pickle.dump(lipophilicity_scaler1, fid)

with open('lipophilicity_scaler2.pkl', 'wb') as fid:

    pickle.dump(lipophilicity_scaler2, fid)

with open('lipophilicity_svr_gb.pkl', 'wb') as fid:

    pickle.dump(lipophilicity_svr_gb, fid)

with open('lipophilicity_xgb_gb.pkl', 'wb') as fid:

    pickle.dump(lipophilicity_xgb_gb, fid)
sns.set(style="whitegrid")

ax = sns.barplot(x=[svr_rmse,nn_rmse,svr_gb_rmse,xgb_gb_rmse],

                 y=['SVR','NN','SVC_GB','XGB_GB'])

ax.set_xlim(0.65,0.8)