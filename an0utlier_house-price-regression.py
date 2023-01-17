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
#data libraries

import pandas as pd

import numpy as np



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)



sns.set_style("whitegrid")
#Reading training data

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



#Reading test data

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



#For submission

submission = pd.DataFrame(columns=["Id", "SalePrice"])

submission["Id"] = df_test["Id"]
with open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', 'r') as f:

    for line in f.readlines() :

        print(line)
df_train.head()
df = df_train.copy()



df.info()
df['SalePrice'].plot(kind = 'hist', bins = 50)
plt.figure(figsize= (10,8))

sns.heatmap(df.corr(), cmap= 'coolwarm')
x = df_train['LotFrontage']

y = df_train['LotArea']

indices = x.between(x.quantile(.05), x.quantile(.95))

plt.figure(figsize=(8,5))

plt.scatter(x[indices],y[indices])
sns.boxplot(x = 'LotConfig', y = 'LotFrontage', data=df_train)
sns.boxplot(x = 'MSZoning', y = 'LotFrontage', data=df)
df['GarageYrBlt'].plot(kind = 'hist', bins = 30)
#Reading training data

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



#Reading test data

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# combining train and test values

data = pd.concat((df_train, df_test)).reset_index(drop=True)

x_saleprice = df_train["SalePrice"]

data.drop(["SalePrice"], axis = 1, inplace= True)



data['PoolQC'].fillna("NA", inplace= True)

data['MiscFeature'].fillna("NA", inplace= True)

data['Alley'].fillna("NA", inplace=True)

data['Fence'].fillna("NA", inplace= True)

data['FireplaceQu'].fillna("NA", inplace= True)

median = data['LotFrontage'].median()

data['LotFrontage'].fillna(median, inplace=True)



data['GarageCond'].fillna('NA', inplace= True)

data['GarageFinish'].fillna('NA', inplace= True)

data['GarageQual'].fillna('NA', inplace= True)

data['GarageType'].fillna('NA', inplace= True)

data['GarageYrBlt'].fillna(0, inplace= True)



data['BsmtExposure'].fillna('NA', inplace= True)

data['BsmtCond'].fillna('NA', inplace= True)

data['BsmtQual'].fillna('NA', inplace= True)

data['BsmtFinType1'].fillna('NA', inplace= True)

data['BsmtFinType2'].fillna('NA', inplace= True)



data['MasVnrType'].fillna('None', inplace= True)

data['MasVnrArea'].fillna(0, inplace= True)



data['MSZoning'].fillna(data['MSZoning'].mode()[0], inplace= True)

data['Functional'].fillna(data['Functional'].mode()[0], inplace= True)

data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mode()[0], inplace= True)

data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0], inplace= True)



data['Utilities'].fillna(data['Utilities'].mode()[0], inplace= True)

data['Electrical'].fillna(data['Electrical'].mode()[0], inplace= True)

data['Exterior1st'].fillna(data['Exterior1st'].mode()[0], inplace= True)

data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0], inplace= True)



data['GarageCars'].fillna(data['GarageCars'].mode()[0], inplace= True)

data['GarageArea'].fillna(data['GarageArea'].mode()[0], inplace= True)

data['KitchenQual'].fillna(data['KitchenQual'].mode()[0], inplace= True)

data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mode()[0], inplace= True)



data['SaleType'].fillna(data['SaleType'].mode()[0], inplace= True)

data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mode()[0], inplace= True)

data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mode()[0], inplace= True)

data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mode()[0], inplace= True)

categorical_columns = data.select_dtypes(include= ['object']).columns

print(categorical_columns)
one_hot_parameters = ['MSZoning' ,'Street', 'Alley', 'LandContour','LotConfig', 'Neighborhood','Condition1', 'Condition2',

                      'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating','GarageType', 

                     'PavedDrive', 'MiscFeature','SaleType', 'SaleCondition' ]
encoders = []



for col in categorical_columns :

    if col not in one_hot_parameters :

        encoders.append(col)
len(encoders) + len(one_hot_parameters) == len(categorical_columns)
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()
for col in encoders :

    data[col] = encoder.fit_transform(data[col].astype(str))
features = pd.get_dummies(data[one_hot_parameters], drop_first= True)
data = pd.concat([data.drop(one_hot_parameters, axis=1), features], axis=1)
train_df = data.iloc[:1460,:]  

train_df['SalePrice'] = x_saleprice

test_df = data.iloc[1460 :,:]  
X_train = train_df.drop('SalePrice', axis=1)

y_train = train_df['SalePrice']

X_test = test_df
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import make_scorer, mean_squared_log_error, r2_score



n_folds = 5



cv = KFold(n_splits= 5, shuffle= True, random_state= 42).get_n_splits(X_train.values)



def test_1(model) :

    msle = make_scorer(mean_squared_log_error)

    rmsle = np.sqrt(cross_val_score(model, X_train, y_train, cv = cv, scoring= msle))

    score_rmsle = [rmsle.mean()]

    return score_rmsle



def test_2(model) :

    r2 = make_scorer(r2_score)

    r2_error = cross_val_score(model, X_train, y_train, cv = cv, scoring= r2)

    score_r2 = [r2_error.mean()]

    return score_r2

import xgboost as xgb



xg_boost = xgb.XGBRegressor(n_estimators= 1000)

test_1(xg_boost)
from sklearn.ensemble import BaggingRegressor



bagging_regressor = BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False,

                                     max_features=1.0, max_samples=1.0, n_estimators=1000,

                                     n_jobs=None, oob_score=False, random_state=51, verbose=0, warm_start=False)



test_1(bagging_regressor)
from sklearn.ensemble import GradientBoostingRegressor



gradient_boosting_reg = GradientBoostingRegressor()



test_1(gradient_boosting_reg)
from sklearn.linear_model import Ridge



ridge = Ridge(alpha=500, copy_X=True, fit_intercept=True, max_iter=None, 

              normalize=False,  random_state=None, solver='auto', tol=0.001)



test_1(ridge)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub_df.info()
gradient_boosting_reg.fit(X_train, y_train)
predictions = gradient_boosting_reg.predict(X_test)
sub_df['SalePrice'] = predictions
sub_df.to_csv('Gradient_Boosting_Regressor.txt', index= False)