import pandas as pd
import numpy as np
import sklearn
import math 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_d = train.copy()
test_d = test.copy()
train_d = pd.get_dummies(train_d)
test_d = pd.get_dummies(test_d)
train_d.head()

keep_cols = train_d.columns
for i in keep_cols:
    if i not in test_d:
        test_d[i] = 0
        
train_d = train_d.fillna(train_d.median())
test_d = test_d.fillna(test_d.median())


train_d['MSSubClass'] = train_d['MSSubClass'].astype('category')
test_d['MSSubClass'] = test_d['MSSubClass'].astype('category')
train_d['OverallQual'] = train_d['OverallQual'].astype('category')
test_d['OverallQual'] = test_d['OverallQual'].astype('category')
train_d['OverallCond'] = train_d['OverallCond'].astype('category')
test_d['OverallCond'] = test_d['OverallCond'].astype('category')
train_d['MoSold'] = train_d['MoSold'].astype('category')
test_d['MoSold'] = test_d['MoSold'].astype('category')
train_d['YrSold'] = train_d['YrSold'].astype('category')
test_d['YrSold'] = test_d['YrSold'].astype('category')
cols = train_d.select_dtypes(include=['number']).columns
train_n = train_d[cols]
test_n = test_d[cols]
train_n.info()
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA 
X = train_d.values
pca = PCA()
pca.fit(X)
X1 = pca.fit_transform(X)

print (len(X1))
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
cv_score = cross_val_score(xgb_test, train_n.drop(['SalePrice','Id'], axis = 1), train_n['SalePrice'], cv = 5, n_jobs = -1)
print('CV Score is: '+ str(np.mean(cv_score)))
