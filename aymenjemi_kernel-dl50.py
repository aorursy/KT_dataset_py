# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn import preprocessing



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn import metrics

from sklearn.metrics import confusion_matrix,accuracy_score





import collections

import tensorflow as tf

from tensorflow.keras import layers



from sklearn.model_selection import StratifiedKFold

from keras.wrappers.scikit_learn import KerasClassifier
test = pd.read_csv('/kaggle/input/dl50-project1/Test.csv')

train = pd.read_csv('/kaggle/input/dl50-project1/Train.csv')
train[train.isna().any(axis=1)]
train.duplicated().sum()
data = train


data.Month.unique()
data.VisitorType.unique()
def get_dummies(df,test = False):

    df.Month = df['Month'].map({'Feb' : 2, 'Mar' : 3, 'May' : 5, 'Oct': 10, 'June' : 6, 'Jul' : 7, 'Aug' : 8, 'Nov' : 11, 'Sep' : 9,'Dec' : 12}).astype(int)

    df.VisitorType = df['VisitorType'].map({'Returning_Visitor' : 2, 'New_Visitor' : 1, 'Other' : 3}).astype(int)

    df.Weekend = df['Weekend'].map( {True: 1, False: 0} ).astype(int)

    if test == False:

        df.Revenue = df['Revenue'].map( {True: 1, False: 0} ).astype(int)
get_dummies(data)
data = data.drop(['id'], axis=1)
data.max()
data.min()
data.std()
def normal_df(df,columns= ['Administrative_Duration' , 'Informational_Duration' , 'ProductRelated' , 'ProductRelated_Duration' , 'PageValues']):



    data_scaler = df[columns]

    scaler = preprocessing.MinMaxScaler()

    std_data = scaler.fit_transform(data_scaler)

    std_data = pd.DataFrame(std_data,columns=columns)

    df[columns] = std_data

    return df
data = normal_df(data)

data.head()
imp = IterativeImputer(random_state=0)

data_clean = imp.fit_transform(data)

data_clean = pd.DataFrame(data_clean , columns =  ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Revenue'])
data_clean[data_clean.isna().any(axis=1)]
X = data_clean.drop(['Revenue'], axis=1)

y = data_clean[['Revenue']]
def create_model_optimizer(optimizer='Nadam'):

    model = tf.keras.Sequential()

    model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal', input_dim=17))

    model.add(layers.Dense(256, activation='relu' , kernel_initializer='random_normal'))

    model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return model



def create_model_loss(loss='binary_crossentropy'):

    model = tf.keras.Sequential()

    model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal', input_dim=17))

    model.add(layers.Dense(256, activation='relu' , kernel_initializer='random_normal'))

    model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss=loss, optimizer='adam',metrics=['accuracy'])

    return model



def create_model_activation(activation='relu'):

    model = tf.keras.Sequential()

    model.add(layers.Dense(128, activation=activation , kernel_initializer='random_normal', input_dim=17))

    model.add(layers.Dense(256, activation=activation , kernel_initializer='random_normal'))

    model.add(layers.Dense(128, activation=activation , kernel_initializer='random_normal'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model



def create_model_kernel_init(init='random_normal'):

    model = tf.keras.Sequential()

    model.add(layers.Dense(128, activation='relu' , kernel_initializer=init, input_dim=17))

    model.add(layers.Dense(256, activation='relu' , kernel_initializer=init))

    model.add(layers.Dense(128, activation='relu' , kernel_initializer=init))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model



def create_model_lstm_optimizer(optimizer='Nadam'):

    clstm = tf.keras.Sequential()

    clstm.add(layers.Embedding(300, 300, input_length=17))

    clstm.add(layers.LSTM(200))

    clstm.add(layers.Dense(1, activation='sigmoid'))

    clstm.summary() 

    clstm.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return clstm



def create_model_lstm_loss(loss='binary_crossentropy'):

    clstm = tf.keras.Sequential()

    clstm.add(layers.Embedding(300, 300, input_length=17))

    clstm.add(layers.LSTM(200))

    clstm.add(layers.Dense(1, activation='sigmoid'))

    clstm.summary() 

    clstm.compile(loss=loss, optimizer='RMSprop',metrics=['accuracy'])

    return clstm



def create_model_lstm_activation(activation='sigmoid'):

    clstm = tf.keras.Sequential()

    clstm.add(layers.Embedding(300, 300, input_length=17))

    clstm.add(layers.LSTM(200))

    clstm.add(layers.Dense(1, activation=activation))

    clstm.summary() 

    clstm.compile(loss='hinge', optimizer='RMSprop',metrics=['accuracy'])

    return clstm
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# cvscores = []

# _X = X.values

# _y = y.values



# for train, test in kfold.split(_X,_y):

#     model = create_model_optimizer()

#     model.fit(_X[train], _y[train],validation_data=(_X[test], _y[test]),batch_size=200,epochs=20,verbose=0)

#     scores = model.evaluate(_X[test], _y[test])

#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#     cvscores.append(scores[1] * 100)



    

# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# cvscores = []



# for train, test in kfold.split(_X,_y):

#     model = create_model_lstm_optimizer()

#     model.fit(_X[train], _y[train],validation_data=(_X[test], _y[test]),batch_size=200,epochs=20,verbose=0)

#     scores = model.evaluate(_X[test], _y[test])

#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#     cvscores.append(scores[1] * 100)



    

# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# model = KerasClassifier(build_fn=create_model_optimizer, epochs=20, batch_size=200, verbose=0)

# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# param_grid = dict(optimizer=optimizer)

# grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

# grid_result = grid.fit(X, y)



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# model = KerasClassifier(build_fn=create_model_loss, epochs=20, batch_size=200, verbose=0)

# loss = ['binary_crossentropy', 'hinge', 'squared_hinge']



# param_grid = dict(loss=loss)

# grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

# grid_result = grid.fit(X, y)



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# model = KerasClassifier(build_fn=create_model_activation, epochs=20, batch_size=200, verbose=0)

# activation = ['sigmoid', 'relu', 'softmax']



# param_grid = dict(activation=activation)

# grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

# grid_result = grid.fit(X, y)



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# create_model_kernel_init

# model = KerasClassifier(build_fn=create_model_kernel_init, epochs=20, batch_size=200, verbose=0)

# init = ['random_normal', 'random_uniform']



# param_grid = dict(init=init)

# grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

# grid_result = grid.fit(X, y)



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# model = KerasClassifier(build_fn=create_model_lstm_optimizer, epochs=20, batch_size=200, verbose=0)

# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# param_grid = dict(optimizer=optimizer)

# grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

# grid_result = grid.fit(X, y)



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# model = KerasClassifier(build_fn=create_model_lstm_loss, epochs=20, batch_size=200, verbose=0)

# loss = ['binary_crossentropy', 'hinge', 'squared_hinge']



# param_grid = dict(loss=loss)

# grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

# grid_result = grid.fit(X, y)



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# model = KerasClassifier(build_fn=create_model_lstm_activation, epochs=20, batch_size=200, verbose=0)

# activation = ['sigmoid', 'relu', 'softmax']



# param_grid = dict(activation=activation)

# grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

# grid_result = grid.fit(X, y)



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
test = pd.read_csv('/kaggle/input/dl50-project1/Test.csv')

data_test = test

data_test = data_test.drop(['id'], axis=1)

get_dummies(data_test,test = True)

data_test = normal_df(data_test)

data_test.head()

model = tf.keras.Sequential()

model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_uniform', input_dim=17))

model.add(layers.Dense(256, activation='relu' , kernel_initializer='random_uniform'))

model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_uniform'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    

model.fit(X, y)

prediction = model.predict(data_test)

submission = test[['id']]

submission['Revenue'] = prediction

submission.to_csv("submission_best.csv", index = False)
model = tf.keras.Sequential()

model.add(layers.Embedding(300, 300, input_length=17))

model.add(layers.LSTM(200))

model.add(layers.Dense(1, activation='sigmoid'))

model.summary() 

model.compile(loss='hinge', optimizer='RMSprop',metrics=['accuracy'])



model.fit(X, y)

prediction = model.predict(data_test)

submission = test[['id']]

submission['Revenue'] = prediction

submission.to_csv("submission_best_lstm.csv", index = False)
submission.shape