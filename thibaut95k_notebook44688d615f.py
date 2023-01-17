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
COL_DESCRIPT = {'MSSubClass': 'numeric',

                'MSZoning': 'categorical',

                'LotFrontage': 'numeric',

                'LotArea': 'numeric',

                'Street': 'categorical',

                'Alley': 'categorical',

                'LotShape': {'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4},

                'LandContour': 'categorical',

                'Utilities': {'AllPub': 1, 'NoSewr': 2, 'NoSeWa': 3, 'ELO': 4},

                'LotConfig': 'categorical',

                'LandSlope': {'Gtl': 1, 'Mod': 2, 'Sev': 3},

                'Neighborhood': 'categorical',

                'Condition1': 'categorical',

                'Condition2': 'categorical',

                'BldgType': 'categorical',

                'HouseStyle': 'categorical',

                'OverallQual': 'numeric',

                'OverallCond': 'numeric',

                'YearBuilt': 'numeric',

                'YearRemodAdd': 'numeric',

                'RoofStyle': 'categorical',

                'RoofMatl': 'categorical',

                'Exterior1st': 'categorical',

                'Exterior2nd': 'categorical',

                'MasVnrType': 'categorical',

                'MasVnrArea': 'numeric',

                'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                 'Foundation': 'categorical',

                'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},

                'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},

                'BsmtFinSF1': 'numeric',

                'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},

                'BsmtFinSF2': 'numeric',

                'BsmtUnfSF': 'numeric',

                'TotalBsmtSF': 'numeric',

                'Heating': 'categorical',

                'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'CentralAir': 'categorical',

                'Electrical': 'categorical',

                '1stFlrSF': 'numeric',

                '2ndFlrSF': 'numeric',

                'LowQualFinSF': 'numeric',

                'GrLivArea': 'numeric',

                'BsmtFullBath': 'numeric',

                'BsmtHalfBath': 'numeric',

                'FullBath': 'numeric',

                'HalfBath': 'numeric',

                'Bedroom': 'numeric',

                'Kitchen': 'numeric',

                'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'TotRmsAbvGrd': 'numeric',

                'Functional': 'categorical',

                'Fireplaces': 'numeric',

                'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'GarageType': 'categorical',

                'GarageYrBlt': 'numeric',

                'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},

                'GarageCars': 'numeric',

                'GarageArea': 'numeric',

                'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                'PavedDrive': {'N': 1, 'P': 2, 'Y': 3},

                'WoodDeckSF': 'numeric',

                'OpenPorchSF': 'numeric',

                'EnclosedPorch': 'numeric',

                '3SsnPorch': 'numeric',

                'ScreenPorch': 'numeric',

                'PoolArea': 'numeric',

                'PoolQC': {'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},

                'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},

                'MiscFeature': 'categorical',

                'MiscVal': 'numeric',

                'MoSold': 'categorical',

                'YrSold': 'numeric',

                'SaleType': 'categorical',

                'SaleCondition': 'categorical'}



def get_ordinal_cols():

    return [c for c in COL_DESCRIPT if isinstance(COL_DESCRIPT[c], dict)]



def handle_ordinal(X):

    for col in get_ordinal_cols():

        X[col] = X[col].map(lambda x: COL_DESCRIPT[col].get(x, 0))

    return X








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



all_data=handle_ordinal(all_data)









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

#del all_data['1stFlrSF']

all_data['TotRmsAbvGrd_sqrt']=np.sqrt(all_data['TotRmsAbvGrd'])

#del all_data['TotRmsAbvGrd']



stringMS = []

for el in np.array(all_data['MSSubClass']):

    stringMS.append(str(el))

   

all_data['MSSubClass'] = stringMS

all_data = pd.get_dummies(all_data)



all_data = all_data.fillna(all_data.mean())
numeric_feats
#all_data.loc[all_data['YearRemodAdd']==all_data['YearBuilt'],'YearRemodAdd']=0

#all_data.loc[all_data['YearRemodAdd']!=0,'YearRemodAdd']=1
import matplotlib.pyplot as plt

plt.plot(X_train['YearRemodAdd'],y, 'ro')
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
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
from sklearn.linear_model import Ridge

model2=Ridge(alpha=10)
def evaluate(model1, model2, X_training, y_training, X_valid, y_valid):

    model1.fit(X_training,  y_training)

    model2.fit(X_training,  y_training)

    preds1 = model1.predict(X_valid)

    preds2 = model2.predict(X_valid)

    preds=(preds1+preds2)/2

    #rmse = math.sqrt(np.sum(np.minimum(abs(preds1-y_valid), abs(preds2-y_valid))*np.minimum(abs(preds1-y_valid), abs(preds2-y_valid))/len(preds1)))

    rmse= math.sqrt(sum((preds-y_valid)*(preds-y_valid)/len(preds)))

    return rmse
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

        res = evaluate(model, model2, X_training, y_training, X_valid, y_valid)

        print(res)

        #cv_ridge = pd.Series(cv_ridge, index = alphas)

        #print(cv_ridge.argmin())

        #res = cv_ridge.min()

        meanRes+=res

    print(meanRes/10)

    return(meanRes/10)
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
#calculateRes()

#BEST 11.55 (2000) (10000: 11.53)
model.fit(X_train,  y)

preds1 = model.predict(X_test)
model2.fit(X_train,  y)

preds2 = model2.predict(X_test)
preds = (preds1+preds2)/2
d = {'Id':test.Id, 'SalePrice':np.exp(preds)}
submit = pd.DataFrame(d)
submit.head()
submit.to_csv('xgBoostRidgeSubmission2.csv', index=False)