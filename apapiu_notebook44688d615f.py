# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pandas import DataFrame, read_csv

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sys

import math



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))







from scipy.stats import skew



#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

numeric_feats = numeric_feats[1:]



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.5]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']

all_data['AgeRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']

all_data['Baths'] = all_data['FullBath'] + all_data['HalfBath']

all_data['BsmtBaths'] = all_data['BsmtFullBath'] + all_data['BsmtHalfBath']

all_data['OverallQual_Square']=all_data['OverallQual']*all_data['OverallQual']

all_data['OverallQual_3']=all_data['OverallQual']*all_data['OverallQual']*all_data['OverallQual']

all_data['OverallQual_exp']=np.exp(all_data['OverallQual'])

all_data['GrLivArea_Square']=all_data['GrLivArea']*all_data['GrLivArea']

all_data['GrLivArea_3']=all_data['GrLivArea']*all_data['GrLivArea']*all_data['GrLivArea']

all_data['GrLivArea_exp']=np.exp(all_data['GrLivArea'])

all_data['GrLivArea_log']=np.log(all_data['GrLivArea'])

all_data['TotalBsmtSF_/GrLivArea']=all_data['TotalBsmtSF']/all_data['GrLivArea']

all_data['OverallCond_sqrt']=np.sqrt(all_data['OverallCond'])

all_data['OverallCond_square']=all_data['OverallCond']*all_data['OverallCond']

all_data['LotArea_sqrt']=np.sqrt(all_data['LotArea'])

all_data['1stFlrSF_sqrt']=np.sqrt(all_data['1stFlrSF'])

del all_data['1stFlrSF']

all_data['TotRmsAbvGrd_sqrt']=np.sqrt(all_data['TotRmsAbvGrd'])

del all_data['TotRmsAbvGrd']



stringMS = []

for el in np.array(all_data['MSSubClass']):

    stringMS.append(str(el))

   

all_data['MSSubClass'] = stringMS

all_data = pd.get_dummies(all_data)



all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
X_training = X_train[200:]

X_valid = X_train[0:200]

y_training = y[200:]

y_valid = y[0:200]
import xgboost

model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0.030,                 

                 learning_rate=0.07,

                 max_depth=5,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.95)
from sklearn.linear_model import Ridge, LassoCV

#model2=Ridge(alpha=10)

model2 = LassoCV(alphas = [1, 0.1, 0.01,0.001, 0.0005, 0.0007, 0.0003])
model2.fit(X_train,  y)
model2.alpha_
calculateRes()

#MSSubclass numerical : 12.37

#MSSubclass categorical :12.29

#MSSubclass categorical, some features removed : 12.52

#MSSubclass categorical, features added : 12.45

#MSSubclass categorical, lots of features added : 12.29

#MSSubclass categorical, lots of features added, maxdepth = 10 : 12.28

#MSSubclass categorical, lots of features added, maxdepth = 5, estimators=1000, gamma=0.030 : 12.04

#best score : 11.96

#best score 2000 est : 11.94

#best score 10000 est : 11.89



#best score 2000 + ridge arithmetic average: 11.64

#best score 10000 + ridge arithmetic average : 11.62



#best score 2000 + geometric average: 11.62
model.fit(X_train,  y)

preds1 = model.predict(X_test)
model2.fit(X_train,  y)

preds2 = model2.predict(X_test)
preds = (preds1+preds2)/2
d = {'Id':test.Id, 'SalePrice':np.exp(preds)}
submit = pd.DataFrame(d)
submit.head()
submit.to_csv('xgBoostRidgeSubmission.csv', index=False)