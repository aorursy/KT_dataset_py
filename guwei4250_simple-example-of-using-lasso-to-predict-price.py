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
import warnings; warnings.simplefilter('ignore')

import scipy.stats as st
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
alldata=pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition'])).reset_index(drop=True)
numeric_features=alldata.select_dtypes(include=[np.number])
skew_feats = numeric_features.apply(lambda x: st.skew(x.dropna()))
skew_feats = skew_feats[skew_feats > 0.75]
alldata[skew_feats.index]=np.log1p(alldata[skew_feats.index])
    
alldata=pd.get_dummies(alldata)
alldata=alldata.fillna(value=0)

trainX = alldata[:train.shape[0]]
testX = alldata[train.shape[0]:]
trainY = np.log1p(train.SalePrice)

testID = test["Id"]
def rmse_cv(model, X, y):
    return np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")))
lasso = LassoCV(alphas = [0.1, 0.01, 0.001, 0.0005])
print(rmse_cv(lasso, trainX, trainY))