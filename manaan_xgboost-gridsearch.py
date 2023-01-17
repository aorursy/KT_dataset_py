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
#importing dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train.describe()
y = train.iloc[:,-1].values

X = train.iloc[:,4:9].values
#visualize

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

plt.ylim(40.6, 40.9)

plt.xlim(-74.1,-73.7)

ax.scatter(train['pickup_longitude'],train['pickup_latitude'], s=0.01, alpha=1)
#essential libs

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import xgboost
#loss func

def rmsle(evaluator,X,real):

    sum = 0.0

    predicted = evaluator.predict(X)

    print("Number predicted less than 0: {}".format(np.where(predicted < 0)[0].shape))



    predicted[predicted < 0] = 0

    for x in range(len(predicted)):

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p-r)**2

    return (sum/len(predicted))**0.5
#find best param for Xgboost

cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

ind_params = {'n_estimators': 100, 'seed':0, 'colsample_bytree': 1, 

             'max_depth': 7, 'min_child_weight': 1}





optimized_GBM = GridSearchCV(xgboost.XGBRegressor(**ind_params), 

                             cv_params,scoring = rmsle, cv =4) 

optimized_GBM.fit(X, np.ravel(y))
optimized_GBM.best_estimator_
optimized_GBM.best_params_
#train on best param

reg = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.01, gamma=0, subsample=0.7,

                           colsample_bytree=1, max_depth=7)



cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)

print(cross_val_score(reg, X, y, cv=cv,scoring=rmsle))

reg.fit(X,y)
test.head()
X_test = test.iloc[:,3:8].values

X_test
#predict

pred = reg.predict(X_test)

pred
#save prediction

pred[pred < 0] = 0

test['trip_duration']=pred.astype(int)

out = test[['id','trip_duration']]

out['trip_duration'].isnull().values.any()

out.to_csv('pred_xgboost.csv',index=False)
test.head()