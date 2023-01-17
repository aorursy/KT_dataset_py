import numpy as np

import pandas as pd

from sklearn.preprocessing import scale

from sklearn import linear_model

from math import sqrt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from scipy.stats import skew
raw_data = pd.read_csv('./train.csv')

test_data = pd.read_csv('./test.csv')
numeric_feat = raw_data.dtypes[raw_data.dtypes!="object"].index

type(numeric_feat)

print(raw_data.shape)

skewed_feats = raw_data[numeric_feat].apply(lambda x: skew(x.dropna()))

skewed_feats = skewed_feats[skewed_feats>0.75]

skewed_feats = skewed_feats.index

raw_data[skewed_feats] = np.log1p(raw_data[skewed_feats])
raw_data.shape
test_data.shape
test_data.isnull().sum()
raw_data.columns


features_pred = ['Id','MSSubClass','LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','MoSold','1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']



features = ['Id','MSSubClass','LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','MoSold','1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr','SalePrice']



#exp = raw_data.head().transpose()

data=raw_data[features].dropna(axis=0);

pred_data = test_data[features_pred].fillna(method='ffill')
pred_data
X = data.loc[:,'MSSubClass':'BedroomAbvGr'].copy()
y = data['SalePrice']

type(y)
X.head()
#finalData = scale(cln_X)

y.head()
X.shape,y.shape
train=X.loc[0:1168,:];

y_train=y[0:1169]
train.shape

#rain.tail()
#y_train.tail()
#crossvalid=X.loc[877:1168]

#y_crossvalid=y[877:1169]

#crossvalid.shape,y_crossvalid.shape
#crossvalid.tail()
#y_crossvalid.tail()
test_data=X.loc[1169:,:]

y_test=y[1169:]

test_data.shape,y_test.shape
finalData=train
#the linearRegrssion model

#reg = linear_model.LinearRegression()
#theElastic net Lasso

l1ratio=0.9

reg=elasticNet  = linear_model.ElasticNet(alpha=0.002,l1_ratio=l1ratio,fit_intercept=True,normalize=True,max_iter=300,warm_start=True)
from sklearn import kernel_ridge
#reg = kernel_ridge.KernelRidge(alpha=0.3, kernel='linear',degree=5, coef0=1)




reg.fit((X),y)
pred=pred_data.loc[:,'MSSubClass':'BedroomAbvGr']

#the origial data

op=reg.predict(pred)

op.shape
op[0:10],y_test[0:10]
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = op))

print(RMSE)
type(op)
id=pred_data.loc[:,'Id']

d = {'Id': id, 'SalePrice': op}

df = pd.DataFrame(data=d,index=None)
df.head()
df.to_csv('opDay2_2.csv', index=False)
df.describe()
df[df['SalePrice']<0]