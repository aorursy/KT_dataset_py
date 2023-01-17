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

BATCH_SIZE = 128

VAL_RATIO = 0.1

TEST_RATIO = 0.1
esol_df= pd.read_csv('../input/esol_descriptors/delaney_df_revised.csv')

print(esol_df.shape)

esol_df.head()
esol_df.drop(['smiles','Compound ID','Molecular Weight',

              'ESOL predicted log solubility in mols per litre','Minimum Degree',

              'Number of H-Bond Donors','Number of Rings',

              'Number of Rotatable Bonds','Polar Surface Area'], axis=1, inplace=True)

print(esol_df.shape)

esol_df.head()
# Get indices of NaN

#inds = pd.isnull(tox21_df).any(1).nonzero()[0]
# Drop NaN from the dataframe

#tox21_df.dropna(inplace=True)

#print(tox21_df.shape)

#tox21_df.head()
esol_df = esol_df.fillna(0)

esol_df.head()
esol_descriptors_df= pd.read_csv('../input/esol_descriptors/delaney_descriptors_df.csv',low_memory=False)

print(esol_descriptors_df.shape)

esol_descriptors_df.head()
# function to coerce all data types to numeric



def coerce_to_numeric(df, column_list):

    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
coerce_to_numeric(esol_descriptors_df, esol_descriptors_df.columns)

esol_descriptors_df.head()
esol_descriptors_df = esol_descriptors_df.fillna(0)

esol_descriptors_df.head()
#tox21_descriptors_df.drop(tox21_descriptors_df.index[inds],inplace=True)

#tox21_descriptors_df.shape
esol_scaler1 = StandardScaler()

esol_scaler1.fit(esol_descriptors_df.values)

esol_descriptors_df = pd.DataFrame(esol_scaler1.transform(esol_descriptors_df.values),

                                   columns=esol_descriptors_df.columns)
nca = NCA1

cn = ['col'+str(x) for x in range(nca)]
esol_transformer1 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)

esol_transformer1.fit(esol_descriptors_df.values)

esol_descriptors_df = pd.DataFrame(esol_transformer1.transform(esol_descriptors_df.values),

                                   columns=cn)

print(esol_descriptors_df.shape)

esol_descriptors_df.head()
X_train, X_test, y_train, y_test = train_test_split(esol_descriptors_df.values,

                                                    esol_df.values, random_state=32,

                                                    test_size=TEST_RATIO)
parameters = {'estimator__kernel':['rbf'], 

              'estimator__epsilon':[0.1,0.25,0.5],

              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}

esol_svr = GridSearchCV(MultiOutputRegressor(SVR()), 

                        parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
result = esol_svr.fit(X_train, y_train)
pred = esol_svr.predict(X_test)

svr_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(svr_rmse)
def huber_loss(y_true, y_pred):

        return tf.losses.huber_loss(y_true,y_pred)
esol_model = Sequential()

esol_model.add(Dense(128, input_dim=esol_descriptors_df.shape[1], 

                     kernel_initializer='he_uniform'))

esol_model.add(BatchNormalization())

esol_model.add(Activation('tanh'))

esol_model.add(Dropout(rate=DROPRATE))

esol_model.add(Dense(64,kernel_initializer='he_uniform'))

esol_model.add(BatchNormalization())

esol_model.add(Activation('tanh'))

esol_model.add(Dropout(rate=DROPRATE))

esol_model.add(Dense(32,kernel_initializer='he_uniform'))

esol_model.add(BatchNormalization())

esol_model.add(Activation('tanh'))

esol_model.add(Dropout(rate=DROPRATE))

esol_model.add(Dense(esol_df.shape[1],kernel_initializer='he_uniform',activation=None))
esol_model.compile(loss=huber_loss, optimizer='adam',metrics=['mse'])
checkpoint = ModelCheckpoint('esol_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
hist = esol_model.fit(X_train, y_train, 

                      validation_split=VAL_RATIO,epochs=EP, batch_size=BATCH_SIZE, 

                      callbacks=[checkpoint])
plt.ylim(0., 1.)

plt.plot(hist.epoch, hist.history["loss"], label="Train loss")

plt.plot(hist.epoch, hist.history["val_loss"], label="Valid loss")
esol_model.load_weights('esol_model.h5')
pred = esol_model.predict(X_test)

nn_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(nn_rmse)
inp = esol_model.input

out = esol_model.layers[-2].output

esol_model_gb = Model(inp, out)
X_train = esol_model_gb.predict(X_train)

X_test = esol_model_gb.predict(X_test)
data = np.concatenate((X_train,X_test),axis=0)
esol_scaler2 = StandardScaler()

esol_scaler2.fit(data)

X_train = esol_scaler2.transform(X_train)

X_test = esol_scaler2.transform(X_test)
data = np.concatenate((X_train,X_test),axis=0)
nca = NCA2
esol_transformer2 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)

esol_transformer2.fit(data)

X_train = esol_transformer2.transform(X_train)

X_test = esol_transformer2.transform(X_test)
nca = X_train.shape[1]

parameters = {'estimator__kernel':['rbf'], 

              'estimator__epsilon':[0.1,0.25,0.5],

              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}

esol_svr_gb = GridSearchCV(MultiOutputRegressor(SVR()), 

                           parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
result = esol_svr_gb.fit(X_train, y_train)
pred = esol_svr_gb.predict(X_test)

svr_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(svr_gb_rmse)
parameters = {'estimator__learning_rate':[0.05,0.1,0.15],'estimator__n_estimators':[75,100,125], 'estimator__max_depth':[3,5,7],

              'estimator__booster':['gbtree','dart'],'estimator__reg_alpha':[0.1,0.05],'estimator__reg_lambda':[0.5,1.]}



esol_xgb_gb = GridSearchCV(MultiOutputRegressor(XGBRegressor(random_state=32)), 

                           parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
result = esol_xgb_gb.fit(X_train, y_train)
pred = esol_xgb_gb.predict(X_test)

xgb_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))

print(xgb_gb_rmse)
with open('esol_svr.pkl', 'wb') as fid:

    pickle.dump(esol_svr, fid)

with open('esol_transformer1.pkl', 'wb') as fid:

    pickle.dump(esol_transformer1, fid)

with open('esol_transformer2.pkl', 'wb') as fid:

    pickle.dump(esol_transformer2, fid)

with open('esol_scaler1.pkl', 'wb') as fid:

    pickle.dump(esol_scaler1, fid)

with open('esol_scaler2.pkl', 'wb') as fid:

    pickle.dump(esol_scaler2, fid)

with open('esol_svr_gb.pkl', 'wb') as fid:

    pickle.dump(esol_svr_gb, fid)

with open('esol_xgb_gb.pkl', 'wb') as fid:

    pickle.dump(esol_xgb_gb, fid)
sns.set(style="whitegrid")

ax = sns.barplot(x=[svr_rmse,nn_rmse,svr_gb_rmse,xgb_gb_rmse],

                 y=['SVR','NN','SVC_GB','XGB_GB'])

ax.set_xlim(0.4,0.75)