import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df=pd.read_csv("../input/train.csv")
sns.regplot(x = 'OverallQual', y = 'SalePrice', data = df, color = 'Blue')
sns.distplot(df['SalePrice'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
plt.figure(figsize = (12, 6))

sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df)

xt = plt.xticks(rotation=45)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np

import xgboost as xgb

import lightgbm as lgb

import time

import matplotlib.pyplot as plt

from sklearn import metrics

#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

allData  = [train , test]

train.head(5)

train.shape
def numericalCol(x):            

     return x.select_dtypes(include=[np.number]).columns.values



# Delete the columns

def colDelete(x,dropColumn):

    x.drop(dropColumn, axis=1, inplace = True)



def colWithNAs(x):            

    z = x.isnull()

    df = np.sum(z, axis = 0)       # Sum vertically, across rows

    col = df[df > 0].index.values 

    return (col)
y = train['SalePrice']  

testIds = test['Id']

train.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')

test.drop( ['Id'], inplace = True, axis = 'columns')  

print(" Sale Price and Id has been dropped from the dataset")

ntrain = train.shape[0]

ntest= test.shape[0]

ntrain,ntest

dropColumn = ['MiscFeature','PoolQC', 'Alley','Fence','FireplaceQu']



for dataset in allData:   

    colDelete(dataset,dropColumn)

    columnsReplaceToNumeric = numericalCol(dataset)

    dataset[columnsReplaceToNumeric]=dataset[columnsReplaceToNumeric].fillna(dataset[columnsReplaceToNumeric].mean(), inplace = True)

    columnWithNAs = colWithNAs(dataset)

    dataset[columnWithNAs] = dataset[columnWithNAs].fillna(value = "other")

  
allDataDF = pd.concat(allData, axis = 'index')

allDataDF.shape 

allDataAllNumeric = pd.get_dummies(allDataDF)

print(allDataAllNumeric.shape)

train = allDataAllNumeric[:ntrain]

test = allDataAllNumeric[ntrain:]

train.shape

test.shape

y_train = y

X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(

                                     train, y_train,

                                     test_size=0.25,

                                     random_state=42

                                     )
def regression(regr,X_test_sparse,y_test_sparse):

    start = time.time()

    regr.fit(X_train_sparse,y_train_sparse)

    end = time.time()

    rf_model_time=(end-start)/60.0

    print("Time taken to model: ", rf_model_time , " minutes" ) 

    

def regressionPlot(regr,X_test_sparse,y_test_sparse,title):

    predictions=regr.predict(X_test_sparse)

    plt.figure(figsize=(10,6))

    plt.scatter(predictions,y_test_sparse,cmap='plasma')

    plt.title(title)

    plt.show()

    

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test_sparse), np.log1p(predictions))))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
regression(lasso,X_test_sparse,y_test_sparse)

regressionPlot(lasso,X_test_sparse,y_test_sparse,"Lasso Model")
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
regression(ENet,X_test_sparse,y_test_sparse)

regressionPlot(ENet,X_test_sparse,y_test_sparse,"Elastic Net Regression")
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
regression(GBoost,X_test_sparse,y_test_sparse)

regressionPlot(GBoost,X_test_sparse,y_test_sparse,"Gradient Boosting Regression")
modelXgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
regression(modelXgb,X_test_sparse,y_test_sparse)

regressionPlot(modelXgb,X_test_sparse,y_test_sparse,"XGBoost")
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
regression(model_lgb,X_test_sparse,y_test_sparse)

regressionPlot(model_lgb,X_test_sparse,y_test_sparse,"LightGBM")