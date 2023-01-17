from sklearn.preprocessing import *
X_train = np.array([[ 1., -1.,  2.], [ 2.,  0.,  0.],[ 0.,  1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)
# 同样的转换实例可以被用与在训练过程中不可见的测试数据:实现和训练数据一致的缩放和移位操作:

X_test = np.array([[ -3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
print(X_test_minmax)

# 可以检查缩放器（scaler）属性，来观察在训练集中学习到的转换操作的基本性质:
print(min_max_scaler.scale_)                             
print(min_max_scaler.min_ )      
scaler = preprocessing.StandardScaler().fit(X_train)
print('scaler: ', scaler)
print('mean : ', scaler.mean_ )                                     
print('scale: ', scaler.scale_ )                                      
print(scaler.transform(X_train) )                          
X_test = [[-1., 1., 0.]]
scaler.transform(X_test)                
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression,BayesianRidge,ElasticNet,ElasticNetCV,LassoCV
from sklearn.decomposition import PCA, TruncatedSVD
from lightgbm import LGBMRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
select_columns = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                  '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 
                  'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
X = train[select_columns]
y = np.log1p(train.SalePrice)
X.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
X_train.shape
numeric_trans = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('ploy', PolynomialFeatures(interaction_only=False)),
    ('scaler', StandardScaler()),
])

lgb = Pipeline([
    ('preprocessor', numeric_trans),
    # ('SelectKBest', SelectKBest(chi2, k=3)),
    ('SVD', TruncatedSVD(n_components=15)),  # 如何拿到这15个特征的名字？ 
    # ('PCA', PCA(n_components=15)) #,SelectPercentile(percentile=0.8))
    ('LGB', LGBMRegressor()),
    #('reg', LogisticRegression())
])


# ------------------训练-预测---------------------------
# lgb.fit(X_train, y_train)
# y_pred = lgb.predict(X_test)
params = {'preprocessor__ploy__degree':[1,2,3],
         'LGB__num_leaves':[7,10,15],}
gd = GridSearchCV(lgb, params,cv=5, return_train_score=False)
gd.fit(X_train, y_train)
gd.predict(X_test)
#print()
print("model score: %.3f" % gd.score(X_test, y_test))
gd.best_params_
gd.cv_results_
