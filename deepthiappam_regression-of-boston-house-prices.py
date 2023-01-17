from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import KFold,cross_val_score

import numpy as np 

import pandas as pd



file = r'../input/housing.csv'

data = pd.read_csv(file,delim_whitespace=True,header=None)

# data.values

dataset = data.values

X = dataset[:,0:13]

y =dataset[:,13]



def my_base_model():

    model = Sequential()

    model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='relu'))

    model.add(Dense(1,kernel_initializer='normal'))

    model.compile(loss= 'mean_squared_error',optimizer= 'adam')

    return model



seed = 7

np.random.seed(seed)

estimator = KerasRegressor(build_fn=my_base_model,nb_epoch=100,batch_size=5, verbose=0)



kfold = KFold(n_splits=10,random_state=seed)

results = cross_val_score(estimator,X,y,cv=kfold)

print("Baseline: %.2f (%.2f) MSE"%(results.mean(),results.std()))
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



file = r'../input/housing.csv'

data = pd.read_csv(file,delim_whitespace=True,header=None)

# data.values

dataset = data.values

X = dataset[:,0:13]

y = dataset[:,13]

def scaled_model():

    model = Sequential()

    model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='relu'))

    model.add(Dense(1,kernel_initializer='normal'))

    model.compile(loss= 'mean_squared_error',optimizer= 'adam')

    return model



seed = 7

np.random.seed(seed)

estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasRegressor(build_fn=scaled_model,nb_epoch = 100,batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10,random_state=seed)

results = cross_val_score(pipeline,X,y,cv=kfold)

print("Baseline: %.2f (%.2f) MSE"%(results.mean(),results.std()))

#13->[13->6]->1



def deeper_model():

    model = Sequential()

    model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='relu'))

    model.add(Dense(6,kernel_initializer='normal',activation='relu'))

    model.add(Dense(1,kernel_initializer='normal'))

    model.compile(loss= 'mean_squared_error',optimizer= 'adam')

    return model



seed = 7

np.random.seed(seed)

estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasRegressor(build_fn=deeper_model,nb_epoch = 100,batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10,random_state=seed)

results = cross_val_score(pipeline,X,y,cv=kfold)

print("Baseline: %.2f (%.2f) MSE"%(results.mean(),results.std()))

#13->[20]->1



def wider_model():

    model = Sequential()

    model.add(Dense(20,input_dim=13,kernel_initializer='normal',activation='relu'))

    model.add(Dense(1,kernel_initializer='normal'))

    model.compile(loss= 'mean_squared_error',optimizer= 'adam')

    return model



seed = 7

np.random.seed(seed)

estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasRegressor(build_fn=wider_model,nb_epoch = 100,batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10,random_state=seed)

results = cross_val_score(pipeline,X,y,cv=kfold)

print("Baseline: %.2f (%.2f) MSE"%(results.mean(),results.std()))
