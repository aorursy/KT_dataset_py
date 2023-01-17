# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sc

import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)

%pylab inline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import sklearn.metrics as metrics

import statsmodels.formula.api as sm
import numpy as np

import pandas as pd

import scipy.stats as stats

import matplotlib.pyplot as plt

%matplotlib inline

import math
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from collections import defaultdict

import time

import gc

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets.samples_generator import make_regression

from sklearn.ensemble.forest import RandomForestRegressor

from sklearn.linear_model.ridge import Ridge

from sklearn.linear_model.stochastic_gradient import SGDRegressor

from sklearn.svm.classes import SVR

from sklearn.utils import shuffle
train = pd.read_csv ("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train
train.info()
train.describe()
train.dtypes
test = pd.read_csv ("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test
test.info()
test.describe()
test.dtypes
pandas_profiling.ProfileReport(train)
pandas_profiling.ProfileReport(test)
numeric_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
train_num = train[numeric_var_names]
train_cat = train[cat_var_names]
# Use a general function that returns multiple values

def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_summary = train_num.apply(lambda x: var_summary(x)).T
num_summary
import numpy as np

for col in train_num.columns:

    percentiles = train_num[col].quantile([0.01,0.99]).values

    train_num[col] = np.clip(train_num[col], percentiles[0], percentiles[1])
#Handling missings - Method2

def Missing_imputation(x):

    x = x.fillna(x.median())

    return x



train_num=train_num.apply(lambda x: Missing_imputation(x))
#Handling missings - Method2

def Cat_Missing_imputation(x):

    x = x.fillna(x.mode())

    return x



train_cat=train_cat.apply(lambda x: Cat_Missing_imputation(x))
# An utility function to create dummy variable

def create_dummies( df, colname ):

    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)

    df = pd.concat([df, col_dummies], axis=1)

    df.drop( colname, axis = 1, inplace = True )

    return df



for c_feature in train_cat.columns:

    train_cat[c_feature] = train_cat[c_feature].astype('category')

    train_cat = create_dummies(train_cat , c_feature )
train_cat.head().T
train_num.head().T
train_new = pd.concat([train_num, train_cat], axis=1)

train_new.head()
pandas_profiling.ProfileReport(train_new)
numeric_var_names=[key for key in dict(train_new.dtypes) if dict(train_new.dtypes)[key] in ['float64', 'int64', 'float32', 'int32','uint8']]

cat_var_names=[key for key in dict(train_new.dtypes) if dict(train_new.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
train_new_num = train_new[numeric_var_names]
train_new_cat = train_new[cat_var_names]
train_new_num.info()
np.log(train_new.SalePrice).hist()
# Distribution of variables

import seaborn as sns

sns.distplot(np.log(train_new.SalePrice))
# correlation matrix (ranges from 1 to -1)

corrm=train_new.corr()

corrm.to_csv('corrm.csv')
# visualize correlation matrix in Seaborn using a heatmap

sns.heatmap(train_new.corr())
features = train_new[train_new.columns.difference( ['SalePrice'] )]

target = train_new['SalePrice']
features.columns
features.shape
target
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import itertools
features.shape
lm = LinearRegression()
# create the RFE model and select 10 attributes

rfe = RFE(lm, n_features_to_select=30)

rfe = rfe.fit(features,target)
rfe.get_support()
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression



import itertools



lm = LinearRegression()



# create the RFE model and select 10 attributes

rfe = RFE(lm, n_features_to_select=30)

rfe = rfe.fit(features,target)
rfe.get_support()
# summarize the selection of the attributes

feature_map = [(i, v) for i, v in itertools.zip_longest(features.columns, rfe.get_support())]
feature_map
#Alternative of capturing the important variables

RFE_features=features.columns[rfe.get_support()]
RFE_features
features1 = features[RFE_features]
features1.head()
# Feature Selection based on importance

from sklearn.feature_selection import f_regression

F_values, p_values  = f_regression(  features1, target )
import itertools

f_reg_results = [(i, v, z) for i, v, z in itertools.zip_longest(features1.columns, F_values,  ['%.3f' % p for p in p_values])]
f_reg_results=pd.DataFrame(f_reg_results, columns=['Variable','F_Value', 'P_Value'])
f_reg_results.sort_values(by=['F_Value'],ascending = False)
f_reg_results.P_Value = pd.to_numeric(f_reg_results.P_Value)
f_reg_results_new=f_reg_results[f_reg_results.P_Value<=0.2]
f_reg_results_new
f_reg_results_new.to_csv("f_reg_results_new.csv")
f_reg_results_new.info()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
features = train_new[train_new.columns.difference( ['SalePrice'] )]

target = train_new['SalePrice']

features_new = SelectKBest(f_classif, k=15).fit(features, target )
features_new.get_support()
features_new.scores_
# summarize the selection of the attributes

import itertools

feature_map = [(i, v) for i, v in itertools.zip_longest(features.columns, features_new.get_support())]

feature_map

#Alternative of capturing the important variables

KBest_features=features.columns[features_new.get_support()]



selected_features_from_KBest = features[KBest_features]
KBest_features
list_vars1 = list(f_reg_results_new.Variable)

list_vars1
all_columns = "+".join(list_vars1)

my_formula = "SalePrice~" + all_columns



print(my_formula)
import statsmodels.formula.api as sm
model = sm.ols('SalePrice~BldgType_Twnhs+BsmtExposure_Gd +Condition2_PosN +ExterQual_Fa +ExterQual_Gd +ExterQual_TA +GarageCond_Fa +GarageCond_Po +GarageCond_TA +GarageQual_Fa +GarageQual_Gd+GarageQual_Po +GarageQual_TA+KitchenQual_Fa +KitchenQual_Gd +KitchenQual_TA +Neighborhood_Crawfor +Neighborhood_NoRidge +Neighborhood_NridgHt +Neighborhood_StoneBr +RoofMatl_CompShg +RoofMatl_WdShake +RoofMatl_WdShngl+Condition2_RRAn+ExterCond_Po+ExterQual_TA+Exterior1st_ImStucc+Exterior2nd_Other+GarageCars+GrLivArea+Neighborhood_NridgHt+OverallQual+SaleCondition_Alloca+SaleCondition_Partial+SaleType_Con+SaleType_New+TotalBsmtSF',data = train_new)
model = model.fit()
model.summary()
print(model.summary())
my_formula = 'SalePrice~BldgType_Twnhs+BsmtExposure_Gd+Condition2_PosN+ExterQual_Fa+GarageCond_Po+GarageQual_Fa +GarageQual_Gd+GarageQual_Po+KitchenQual_Fa+Neighborhood_Crawfor +Neighborhood_NoRidge +Neighborhood_NridgHt +Neighborhood_StoneBr +RoofMatl_CompShg +RoofMatl_WdShake +RoofMatl_WdShngl+Condition2_RRAn+ExterCond_Po+Exterior1st_ImStucc+Exterior2nd_Other+GarageCars+GrLivArea+Neighborhood_NridgHt+SaleCondition_Alloca+SaleType_Con+SaleType_New+TotalBsmtSF'

my_formula
from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices
# get y and X dataframes based on this regression

y, X = dmatrices(my_formula,train_new,return_type='dataframe')
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.round(1)
Train, Test = train_test_split( train_new, test_size = 0.3, random_state = 123456 )
print(len(Train))

print(len(Test))
import statsmodels.formula.api as smf
my_formula = 'SalePrice~BldgType_Twnhs+BsmtExposure_Gd+Condition2_PosN+ExterQual_Fa+GarageCond_Po+GarageQual_Fa+GarageQual_Gd+GarageQual_Po+KitchenQual_Fa+Neighborhood_Crawfor+Neighborhood_NoRidge+Neighborhood_NridgHt+Neighborhood_StoneBr+RoofMatl_CompShg+RoofMatl_WdShake+RoofMatl_WdShngl+Condition2_RRAn+ExterCond_Po+Exterior1st_ImStucc+Exterior2nd_Other+GarageCars+GrLivArea+SaleCondition_Alloca+SaleType_Con+SaleType_New+TotalBsmtSF'

my_formula
model = smf.ols(my_formula, data=Train).fit()

print(model.summary())
Train['pred'] = pd.DataFrame(model.predict(Train))
Train.head()
Test['pred'] = pd.DataFrame(model.predict(Test))

Test.head()
# calculate these metrics by hand!

from sklearn import metrics

import numpy as np

import scipy.stats as stats
#Train Data

MAPE_Train = np.mean(np.abs(Train.SalePrice - Train.pred)/Train.SalePrice)

print(MAPE_Train)





RMSE_Train = metrics.mean_squared_error(Train.SalePrice ,Train.pred)

print(RMSE_Train)



Corr_Train = stats.stats.pearsonr(Train.SalePrice,Train.pred)

print(Corr_Train)





#Test Data

MAPE_Test = np.mean(np.abs(Test.SalePrice - Test.pred)/Test.SalePrice)

print(MAPE_Test)



RMSE_Test = metrics.mean_squared_error(Test.SalePrice, Test.pred)

print(RMSE_Test)



Corr_Test = stats.stats.pearsonr(Test.SalePrice, Test.pred)

print(Corr_Test)
model.resid.hist(bins=100)
#Decile analysis - Train



Train['Deciles']=pd.qcut(Train['pred'],10, labels=False)



avg_actual = Train[['Deciles','SalePrice']].groupby(Train.Deciles).mean().sort_index(ascending=False)['SalePrice']

avg_pred = Train[['Deciles','pred']].groupby(Train.Deciles).mean().sort_index(ascending=False)['pred']



Decile_analysis_Train = pd.concat([avg_actual, avg_pred], axis=1)



Decile_analysis_Train
#Decile analysis - Train

Test['Deciles']=pd.qcut(Test['pred'],10, labels=False)



avg_actual_Test = Test[['Deciles','SalePrice']].groupby(Test.Deciles).mean().sort_index(ascending=False)['SalePrice']

avg_pred_Test = Test[['Deciles','pred']].groupby(Test.Deciles).mean().sort_index(ascending=False)['pred']



Decile_analysis_Test = pd.concat([avg_actual_Test, avg_pred_Test], axis=1)



Decile_analysis_Test
train_new
from sklearn import model_selection

def split(df):

    train_new = df

    train_new["kfold"] = -1

    train_new = train_new.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=5, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=train_new)):

            print(len(train_idx), len(val_idx))

            train_new.loc[val_idx, 'kfold'] = fold

    return train_new
train_new
train_folds = split(train_new)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Ridge,ElasticNet,Lasso

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb

import lightgbm as lgb



for FOLD in [0,1,2,3,4]:

    print("The Fold is :",FOLD)

    FOLD_MAPPPING = {

        0: [1, 2, 3, 4],

        1: [0, 2, 3, 4],

        2: [0, 1, 3, 4],

        3: [0, 1, 2, 4],

        4: [0, 1, 2, 3]

    }



    train_new = train_folds[train_folds.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)

    valid_new = train_folds[train_folds.kfold==FOLD].reset_index(drop=True)



    ytrain = train_new.SalePrice.values

    yvalid = valid_new.SalePrice.values

    train_new = train_new.drop(["kfold"], axis=1)

    valid_new = valid_new.drop(["kfold"], axis=1)



    valid_new = valid_new[train_new.columns]





    def rmse_cv(model):

        rmse= np.sqrt(-cross_val_score(model, train_new, ytrain, scoring="neg_mean_squared_error", cv = 5))

        return(rmse)



    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                                 learning_rate=0.05, max_depth=3, 

                                 min_child_weight=1.7817, n_estimators=2200,

                                 reg_alpha=0.4640, reg_lambda=0.8571,

                                 subsample=0.5213, silent=1,

                                 random_state =7, nthread = -1)

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



    score = rmse_cv(lasso)

    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmse_cv(ENet)

    print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmse_cv(KRR)

    print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmse_cv(GBoost)

    print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmse_cv(model_xgb)

    print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmse_cv(model_lgb)

    print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

    

    

    print(train_new.shape)