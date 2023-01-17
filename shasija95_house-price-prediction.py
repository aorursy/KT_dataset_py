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
import pandas as pd

import numpy as np

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.head()
y_train = data_train['SalePrice']

#price.head()

data_train.drop('SalePrice', axis=1, inplace = True)

data_train.head()

train_n = data_train.shape[0]

test_id = data_test['Id']
#data_test.head()

all_data = pd.concat((data_train,data_test))

all_data = all_data.reset_index(drop=True)

all_data.drop('Id', axis=True, inplace=True)

all_data.head()

#data_var = data.iloc[:,:-1]

#data_var.head()
corr_mat = all_data.corr()

data = go.Heatmap(z=corr_mat, x=all_data.columns.values, y=all_data.columns.values)

#label = go.Layout(xaxis=all_data.columns, yaxis=all_data.columns )

#fig = go.Figure([data],label)

iplot([data])
#pd.get_dummies(data_var, columns=['MSZoning']).head()

all_data.columns
missing_vals = all_data.isnull().sum().sort_values(ascending=False)

data = go.Bar(y=missing_vals.values[:20], x = missing_vals.index[:20])

iplot([data])
all_data['PoolQC'].fillna(value='None', inplace=True)
all_data['MiscFeature'].fillna(value='None', inplace=True)
all_data['Alley'].fillna(value='None', inplace=True)
all_data['Fence'].fillna(value='None', inplace=True)
all_data['FireplaceQu'].fillna(value='None', inplace=True)
median = all_data['LotFrontage'].median()

all_data['LotFrontage'].fillna(value=median, inplace=True)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
missing_vals[missing_vals>0]
for _ in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[_].fillna(value=0, inplace=True)
all_data[all_data['Utilities']!='AllPub']['Utilities']
for _ in ['BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']:

    all_data[_].fillna(value='None', inplace=True)
for _ in ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF']:

    all_data[_].fillna(value=0, inplace=True)
all_data['MasVnrType'].fillna(value='None', inplace=True)
all_data['MasVnrArea'].fillna(value=0, inplace=True)
mode = all_data['MSZoning'].mode()[0]

all_data['MSZoning'].fillna(value=mode, inplace=True)
#as most all of the values are allpubs and in trainning set and test setexect one in trainnig set, we will frop this feature

all_data.drop(labels='Utilities', axis=1, inplace=True)
mode = all_data['Functional'].mode()[0]

all_data['Functional'].fillna(value=mode, inplace=True)

#all_data[all_data['Functional'].isnull()]
mode = all_data['Exterior2nd'].mode()[0]

all_data['Exterior2nd'].fillna(value=mode, inplace=True)
mode = all_data['Exterior1st'].mode()[0]

all_data['Exterior1st'].fillna(value=mode, inplace=True)
mode = all_data['SaleType'].mode()[0]

all_data['SaleType'].fillna(value=mode, inplace=True)
mode = all_data['Electrical'].mode()[0]

all_data['Electrical'].fillna(value=mode, inplace=True)
mode = all_data['KitchenQual'].mode()[0]

all_data['KitchenQual'].fillna(value=mode, inplace=True)
#NO more NaN values

xx=all_data.isnull().sum().sort_values(ascending=False)

xx[xx>0]

'OverallQual', 'OverallCond'
from scipy.stats import skew

numics = all_data.select_dtypes(include=['float64', 'int64']).columns

skewness = all_data[numics].apply(lambda x : skew(x.dropna())).sort_values(ascending=False)

skew_feat = abs(skewness[abs(skewness)>1]).sort_values(ascending=False).index

from scipy.special import boxcox1p

lam = 0.15

for _ in skew_feat:

    all_data[_] = boxcox1p(all_data[_], lam)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

all_data_old = all_data



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

       'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

for c in cols:

    le.fit(all_data[c])

    r=le.transform(all_data[c])

    all_data[c]=r

all_data = pd.get_dummies(all_data)




## Seperating training and testing data

train_n

train_new = all_data[:train_n]

test_new = all_data[train_n:]
## Appling Decision tree and linear regressor for prediction.

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(min_samples_split=10)

dtr.fit(train_new, y_train)
dtr.score(train_new, y_train)

from sklearn.linear_model import Ridge

lr = Ridge()

lr.fit(train_new, y_train)

lr.score(train_new, y_train)

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

n_estimators = [100, 200, 300, 400, 500]

learning_rate = [0.001, 0.01, 0.1, 0.2]

param = dict(n_estimators=n_estimators, learning_rate=learning_rate)

xgbr = xgb.XGBRegressor()

rgs = GridSearchCV(xgbr, param)

rgs.fit(train_new, y_train)
from sklearn.base import clone

from sklearn.model_selection import KFold

import numpy as np

class StackedRegression():

    def __init__(self, base_model, meta_model, kfold=5):

        self.base_model=base_model

        self.meta_model=meta_model

        self.kfold=kfold

    def fit(self, X, y):

        train_meta_model = np.zeros((X.shape[0], len(self.base_model)))

        self.base_models_ = [list() for _ in self.base_model]

        kf = KFold(n_splits=self.kfold)

        #spliting = kf.split(X,y)

        for i, model in enumerate(self.base_model):

            print ("Printing model ")

            print (model)

            #self.base_models_ = [list() for _ in self.base_model]

            #kf = KFold(n_splits=self.kfold)

            for k, l in kf.split(X,y):

                cloned_model = clone(model)

                #print (X.shape, y.shape)

                cloned_model.fit(X[k],y[k])

                print (train_meta_model.shape)

                train_meta_model[l,i] = cloned_model.predict(X[l])

                self.base_models_[i].append(cloned_model)

        self.meta_model_ = clone(self.meta_model)

        self.meta_model_.fit(train_meta_model, y)

        return self

    

    def predict(self, x):

        gg=[models_ for models_ in self.base_models_]

        #print (self.base_models_)

        meta_test = np.column_stack([np.column_stack([similar.predict(x) for similar in models_]).mean(axis=1) for models_ in self.base_models_])

        return self.meta_model_.predict(meta_test)

        

        

                        

                

                
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import GradientBoostingRegressor

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

xgbr2=xgb.XGBRegressor(learning_rate=0.05,  n_estimators=2200)

gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05)



sr = StackedRegression(base_model=[KRR, gb, ENet], meta_model=lasso)
sr.fit(train_new.values, y_train)

#out=sr.predict(test_new)

#out
out=sr.predict(test_new.values)

out = out

#sub=out
from sklearn.metrics import mean_squared_error

stacked_train = sr.predict(train_new.values)

#stacked_train

xgbr2.fit(train_new.values,y_train)

xgboost_test = xgbr2.predict(test_new.values)

import lightgbm as lgb

lgbm = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=720)

lgbm.fit(train_new.values, y_train)

lgbm_y = lgbm.predict(test_new.values)

ensemble = np.expm1(out)*0.70 + np.expm1(xgboost_test)*0.15 + np.expm1(lgbm_y)*0.15
sub=ensemble

sub
#from sklearn.model_selection import cross_val_score

#acc= cross_val_score(xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

           #                  learning_rate=0.05, max_depth=3, 

            #                 min_child_weight=1.7817, n_estimators=2200,

             #                reg_alpha=0.4640, reg_lambda=0.8571,

              #               subsample=0.5213, silent=1,

               #              random_state =7, nthread = -1), scoring="neg_mean_squared_error", X=train_new, y=y_train, cv=10)

#sub = rgs.predict(test_new)

#sub

#rgs.best_params_

#sub.shape
#sub = lr.predict(test_new)

#sub

#acc1 = np.sqrt(-acc)

#acc1.mean()
df = pd.DataFrame()

df['Id']=test_id

df['SalePrice']=sub

df.to_csv('submission.csv',index=False)