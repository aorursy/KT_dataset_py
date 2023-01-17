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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as stats

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE, SelectKBest, f_regression

import sklearn.metrics as metrics

from sklearn.linear_model import LinearRegression   #Linear Regression using sklearn



import statsmodels.formula.api as smf               #Linear Regression using statsmodels



import pandas_profiling                  
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
data.drop(["Id",'MiscFeature','Fence','PoolQC','FireplaceQu','Alley'], axis=1, inplace = True)
numeric_var_names=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
data_cat = data[cat_var_names]

data_num = data[numeric_var_names]
def outlier_miss_treat(x):

    x = x.clip(upper= x.quantile(0.99))

    x = x.clip(lower= x.quantile(0.01))

    x = x.fillna(x.median())

    return x
data_num_new = data_num.apply(outlier_miss_treat)
def miss_treat_cat(x):

    x= x.fillna("blank")

    x= x.fillna(x.mode())

    return x
data_cat_new = data_cat.apply(miss_treat_cat)
cat_dummies = pd.get_dummies(data_cat_new, drop_first=True)
data_new = pd.concat([data_num_new, cat_dummies], axis=1)
np.log(data_new.SalePrice).hist(bins=20)
data_new['ln_SalePrice'] =  np.log(data_new.SalePrice)
feature_cols  =['OverallQual',

'GrLivArea',

'TotalBsmtSF',

'GarageCars',

'1stFlrSF',

'YearBuilt',

'BsmtFinSF1',

'OverallCond',

'LotArea',

'KitchenAbvGr',

'CentralAir_Y',

'BsmtUnfSF',

'GarageArea',

'GarageType_Attchd',

'YearRemodAdd',

'OpenPorchSF',

'BsmtQual_Gd',

'ExterCond_Fa',

'GarageYrBlt',

'2ndFlrSF',

'LotFrontage',

'GarageQual_TA',

'MasVnrType_Stone',

'BedroomAbvGr',

'BsmtFullBath',

'GarageType_Detchd',

'MoSold',

'HalfBath',

'KitchenQual_Gd',

'BsmtExposure_No',

'SaleType_ConLD',

'WoodDeckSF',

'ExterCond_TA',

'Exterior1st_VinylSd',

'YrSold',

'KitchenQual_TA',

'GarageType_BuiltIn',

'SaleType_WD',

'BsmtFinType1_Unf',

'Exterior2nd_VinylSd',

'LandSlope_Mod',

'Exterior2nd_Wd Sdng',

'MasVnrArea'

]



X = data_new[feature_cols]

y = data_new['ln_SalePrice']
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)

treereg
from sklearn.model_selection  import cross_val_score

scores = cross_val_score(treereg, X, y, cv=14, scoring='neg_mean_squared_error')



np.mean(np.sqrt(-scores))
treereg = DecisionTreeRegressor(max_depth=1, random_state=1)



scores = cross_val_score(treereg, X, y, cv=14, scoring='neg_mean_squared_error')

np.mean(np.sqrt(-scores))
max_depth_range = range(1, 9)



# list to store the average RMSE for each value of max_depth

RMSE_scores = []



# use LOOCV with each value of max_depth

for depth in max_depth_range:

    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)

    MSE_scores = cross_val_score(treereg, X, y, cv=14, scoring='neg_mean_squared_error')

    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
# plot max_depth (x-axis) versus RMSE (y-axis)

plt.plot(max_depth_range, RMSE_scores)

plt.xlabel('max_depth')

plt.ylabel('RMSE (lower is better)')
# max_depth=6 was best, so fit a tree using that parameter

treereg = DecisionTreeRegressor(max_depth=6, random_state=1)

treereg.fit(X, y)
treereg.feature_importances_
# "Gini importance" of each feature: the (normalized) total reduction of error brought by that feature

data1 = pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})
data1.to_csv('features.csv')
# use fitted model to make predictions on testing data

X_test = data_new[feature_cols]

y_test = data_new['SalePrice']



y_pred = np.exp(treereg.predict(X_test))

y_pred
# calculate RMSE

np.sqrt(metrics.mean_squared_error(y_test, y_pred))
data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
indices = data_test.index
data_test.drop(['MiscFeature','Fence','PoolQC','FireplaceQu','Alley'], axis=1, inplace = True)



numeric_var_names=[key for key in dict(data_test.dtypes) if dict(data_test.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(data_test.dtypes) if dict(data_test.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)



data_cat = data_test[cat_var_names]

data_num = data_test[numeric_var_names]



def outlier_miss_treat(x):

    x = x.clip(upper= x.quantile(0.99))

    x = x.clip(lower= x.quantile(0.01))

    x = x.fillna(x.median())

    return x



data_num_new = data_num.apply(outlier_miss_treat)



def miss_treat_cat(x):

    x= x.fillna("blank")

    x= x.fillna(x.mode())

    return x



data_cat_new = data_cat.apply(miss_treat_cat)



cat_dummies = pd.get_dummies(data_cat_new, drop_first=True)



data_new = pd.concat([data_num_new, cat_dummies], axis=1)
preds = treereg.predict(data_new[feature_cols]).ravel()

output = pd.DataFrame({"Id": indices,"ln_SalesPrice": preds})

output

output['SalePrice'] = np.exp(output['ln_SalesPrice'])

output.drop(["ln_SalesPrice"],axis=1, inplace= True)
output.to_csv('submission.csv', index=False)