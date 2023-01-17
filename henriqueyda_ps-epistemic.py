%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/epistemicselecao/regression_features_train.csv")

train_label = pd.read_csv("../input/epistemicselecao/regression_targets_train.csv")

test_data = pd.read_csv("../input/epistemicselecao/regression_features_test.csv")

test_label = pd.read_csv("../input/epistemicselecao/regression_targets_test.csv")
train_data = pd.get_dummies(train_data, columns=['cbwd'])
x = 0

Xtrain = pd.DataFrame() 

# Da cidade 0 a cidade 4

for i in range(0,5):

    #Do ano 2010 a 2013

    for j in range(2010,2014):

        #Do mês 1 ao mês 12

        for k in range(1,13):

            #Do dia 1 ao dia 31

            for l in range(1,32):

                Xtrain[x] = train_data[(train_data['day'] == l) & (train_data['year'] == j) & (train_data['month'] == k) & (train_data['city'] == i)].mean()

                x = x+1
Xtrain = Xtrain.T

Xtrain = Xtrain.dropna()

Xtrain = Xtrain.reset_index(drop=True)
Ytrain = Xtrain['precipitation']

Xtrain = Xtrain.drop(columns=['precipitation'])
Xtrain
test_data = pd.get_dummies(test_data, columns=['cbwd'])

x = 0

Xtest = pd.DataFrame() 

for i in range(0,5):

    for j in range(2014,2016):

        for k in range(1,13):

            for l in range(1,32):

                Xtest[x] = test_data[(test_data['day'] == l) & (test_data['year'] == j) & (test_data['month'] == k) & (test_data['city'] == i)].mean()

                x = x+1
Xtest = Xtest.T

Xtest = Xtest.dropna()

Xtest = Xtest.reset_index(drop=True)
x = 0

Ytest = pd.DataFrame() 

for i in range(0,5):

    for j in range(2014,2016):

        for k in range(1,13):

            for l in range(1,32):

                Ytest[x] = test_data[(test_data['day'] == l) & (test_data['year'] == j) & (test_data['month'] == k) & (test_data['city'] == i)].mean()

                x = x+1
Ytest = Ytest.T

Ytest = Ytest.dropna()

Ytest = Ytest.reset_index(drop=True)
Xtrain.info()
Xtrain.describe()
Xtrain.hist(bins=30, figsize=(20,15))

plt.show()
Xtrain_aux = Xtrain

Xtrain_aux['precipitation'] = Ytrain

corr_matrix = Xtrain_aux.corr()

corr_matrix
Xtrain = Xtrain[['season','DEWP', 'HUMI','PRES', 'TEMP','city', 'cbwd_NE', 'cbwd_NW','cbwd_SE', 'cbwd_SW', 'cbwd_cv']]

Xtest = Xtest[['season', 'DEWP', 'HUMI','PRES', 'TEMP', 'city', 'cbwd_NE', 'cbwd_NW','cbwd_SE', 'cbwd_SW', 'cbwd_cv']]
from keras import Sequential

from keras.layers import Dense

from keras.optimizers import Adam
model = Sequential()
mean = Xtrain.mean(axis=0)

Xtrain -= mean

std = Xtrain.std(axis=0)

Xtrain /= std
mean = Xtest.mean(axis=0)

Xtest -= mean

std = Xtest.std(axis=0)

Xtest /= std
model.add(Dense(16, activation='relu', input_shape=(Xtrain.shape[1],)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1))
adam = Adam(lr=0.01)
model.compile(optimizer=adam, loss='mae', metrics=['mse'])
model.fit(Xtrain, Ytrain, epochs=50, batch_size = 16)
model.predict(Xtest)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=25, max_depth = 20)

rf.fit(Xtrain, Ytrain)
predict = rf.predict(Xtest)
Ytest['precipitation'] = predict
Ytest
train_class_data = pd.read_csv("../input/epistemicselecao/classification_features_train.csv")

train_class_label = pd.read_csv("../input/epistemicselecao/classification_targets_train.csv")

test_class_data = pd.read_csv("../input/epistemicselecao/classification_features_test.csv")

test_class_label = pd.read_csv("../input/epistemicselecao/classification_targets_test.csv")
train_class_data = train_class_data.loc[:, train_class_data.columns != 'precipitation']

train_class_label = train_class_label.sort_values(['city', 'year'], ascending=[True, True])

train_class_label = train_class_label.reset_index(drop=True)

train_class_data = pd.get_dummies(train_class_data, columns=['cbwd'])
x = 0

Xclasstrain = pd.DataFrame() 

for i in range(0,5):

    for j in range(2010,2014):

        for k in range(1,13):

            for l in range(1,32):

                Xclasstrain[x] = train_class_data[(train_class_data['day'] == l) & (train_class_data['year'] == j) & (train_class_data['month'] == k) & (train_class_data['city'] == i)].mean()

                x = x+1
Xclasstrain = Xclasstrain.T

Xclasstrain = Xclasstrain.dropna()

Xclasstrain = Xclasstrain.reset_index(drop=True)
test_class_label = test_class_label.sort_values(['city', 'year'], ascending=[True, True])

test_class_label = test_class_label.reset_index(drop=True)

test_class_data = pd.get_dummies(test_class_data, columns=['cbwd'])
x = 0

Xclasstest = pd.DataFrame() 

for i in range(0,5):

    for j in range(2014,2016):

        for k in range(1,13):

            for l in range(1,32):

                Xclasstest[x] = test_class_data[(test_class_data['day'] == l) & (test_class_data['year'] == j) & (test_class_data['month'] == k) & (test_class_data['city'] == i)].mean()

                x = x+1
Xclasstest = Xclasstest.T

Xclasstest = Xclasstest.dropna()

Xclasstest = Xclasstest.reset_index(drop=True)
Xclasstest
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
log_reg = LogisticRegression(solver='lbfgs', max_iter =100000)
Yclasstrain = train_class_label['rain']
Yclasstest = test_class_label['rain']
log_reg.fit(Xclasstrain, Yclasstrain)
lr_probs = log_reg.predict_proba(Xclasstest)
lr_probs
lr_probs = lr_probs[:, 1]

lr_probs