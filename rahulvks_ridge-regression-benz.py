import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import model_selection , preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import random







train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train_File_Size", train.shape)

print("Test_File_Size", test.shape)
plt.figure(figsize=(4,6))

plt.scatter(range(train.shape[0]), np.sort(train.y.values))
for f in train.columns:

    if train[f].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values))

        train[f] = lbl.transform(list(train[f].values))

        

for f in test.columns:

    if test[f].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(test[f].values))

        test[f] = lbl.transform(list(test[f].values))       

        
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))



#train_scaled = min_max_scaler.fit_transform([np.float32(train)])
id_test = test.ID

x_target = train['y']

x_train = train.drop(['y','ID'],axis=1)

x_test = test.drop(['ID'],axis=1)
from sklearn.linear_model import Ridge

clf = Ridge()

ridge_params = {'alpha': [0,0.5,1,2,3,5]}

ridge_grid = model_selection.GridSearchCV(clf,ridge_params,cv=5,verbose=10,scoring='r2')
model = ridge_grid.fit(x_train, x_target) 
pred = model.predict(x_test)
sub = pd.DataFrame()

sub['ID'] = id_test

sub['y'] = pred

sub.to_csv('ridge.csv', index=False)
seed= 7

model = Ridge()

kfold = model_selection.KFold(n_splits=10,random_state=seed)

scoring = 'r2'

result = model_selection.cross_val_score(model, x_train,x_target, cv=kfold, scoring=scoring)



scoring = 'neg_mean_squared_error'

results = model_selection.cross_val_score(model, x_train, x_target, cv=kfold, scoring=scoring)

#print("R^2: %.3f (%.3f)") % (result.mean(), result.std())