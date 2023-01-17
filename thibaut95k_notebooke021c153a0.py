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
from sklearn.linear_model import Ridge

model = Ridge(alpha=10)

def calculatePreds(model):

    preds = np.zeros(0)

    for i in range(0, 10):

        X_training = pd.concat([X_train[0:i*146], X_train[(i+1)*146:1460]])

        X_valid = X_train[i*146:(i+1)*146]

        y_training = pd.concat([y[0:i*146], y[(i+1)*146:1460]])

        y_valid = y[i*146:(i+1)*146]

        alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

        #cv_ridge = [rmse_cv(Ridge(alpha =alpha), X_training, y_training, X_valid, y_valid)

        #            for alpha in alphas]

        preds = np.concatenate((preds,getpreds(model, X_training, y_training, X_valid, y_valid)), axis=0)

        #cv_ridge = pd.Series(cv_ridge, index = alphas)

        #print(cv_ridge.argmin())

        #res = cv_ridge.min()

    return(preds)





def getpreds(model, X_training, y_training, X_valid, y_valid):

    model.fit(X_training,  y_training)

    preds = model.predict(X_valid)

    return(preds)
preds = calculatePreds(model)
X_train['PredsRidge']=preds
import xgboost

model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0.030,                 

                 learning_rate=0.07,

                 max_depth=5,

                 min_child_weight=1.5,

                 n_estimators=2000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.95)
def calculateRes():

    meanRes=0

    for i in range(0, 10):

        X_training = pd.concat([X_train[0:i*146], X_train[(i+1)*146:1460]])

        X_valid = X_train[i*146:(i+1)*146]

        y_training = pd.concat([y[0:i*146], y[(i+1)*146:1460]])

        y_valid = y[i*146:(i+1)*146]

        alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

        #cv_ridge = [rmse_cv(Ridge(alpha =alpha), X_training, y_training, X_valid, y_valid)

        #            for alpha in alphas]

        res = evaluate(model, X_training, y_training, X_valid, y_valid)

        print(res)

        #cv_ridge = pd.Series(cv_ridge, index = alphas)

        #print(cv_ridge.argmin())

        #res = cv_ridge.min()

        meanRes+=res

    print(meanRes/10)

    return(meanRes/10)
def evaluate(model, X_training, y_training, X_valid, y_valid):

    model.fit(X_training,  y_training)

    preds = model.predict(X_valid)

    rmse= math.sqrt(sum((preds-y_valid)*(preds-y_valid)/len(preds)))

    return rmse
calculateRes()