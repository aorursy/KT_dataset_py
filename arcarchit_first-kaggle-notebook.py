# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv('../input/train.csv');

testDF = pd.read_csv('../input/test.csv');
all_data = pd.concat((trainDF.loc[:,'MSSubClass':'SaleCondition'],

                      testDF.loc[:,'MSSubClass':'SaleCondition']))

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:

X_train = all_data[:trainDF.shape[0]]

X_test = all_data[trainDF.shape[0]:]

y = trainDF.SalePrice
# Sample submission

#submission = pd.DataFrame({ 'Id': testDF['Id'],

 #                           'SalePrice': yt })

#submission.to_csv("submission.csv", index=False)
Xtr, Xte, Ytr, Yte = train_test_split(X_train, y, test_size=0.33, random_state=42)
import xgboost as xgb
T_train_xgb = xgb.DMatrix(Xtr, Ytr)
params = {"objective": "reg:linear"}

gbm = xgb.train(dtrain=T_train_xgb,params=params)
from sklearn.model_selection import cross_val_score
def r2_cv(model):

    m= np.mean(cross_val_score(model, X_train, y, scoring="r2", cv = 5))

    return(m)
Y_pred = gbm.predict(xgb.DMatrix(Xte))
def error(actual, predicted):

    actual = np.log(actual)

    predicted = np.log(predicted)

    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))
def invboxcox(y,ld):

   if ld == 0:

      return(np.exp(y))

   else:

      return(np.exp(np.log(ld*y+1)/ld))
def my_cv(model, x, y, n_splits, boxcox_lambda):

    kf = KFold(n_splits=n_splits)

    kf.get_n_splits(X_train)

    ary = []

    for train_index, test_index in kf.split(x):       

        Xtr, Xte = X_train.ix[train_index, :], X_train.ix[test_index, :]

        Ytr, Yte = y[train_index], y[test_index]

        Ytr = stats.boxcox(Ytr,boxcox_lambda) if boxcox_lambda!=None else Ytr

        model.fit(Xtr, Ytr)

        Y_pred = model.predict(Xte)

        Y_pred = invboxcox(Y_pred, boxcox_lambda) if boxcox_lambda!=None else Y_pred        

        ary.append(error(Yte, Y_pred))

    return np.mean(ary)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

kf.get_n_splits(X_train)

ary = []

for train_index, test_index in kf.split(X_train):       

    Xte =  X_train.ix[test_index, :]

    Xtr = X_train.ix[train_index, :]    

    Ytr, Yte = y[train_index], y[test_index]

    T_train_xgb = xgb.DMatrix(Xtr, Ytr)

    gbm = xgb.train(dtrain=T_train_xgb,params=params)

    Y_pred = gbm.predict(xgb.DMatrix(Xte))

    ary.append(error(Yte, Y_pred))

print (np.mean(ary))