# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns # prettier plotting

%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# loading the data
train_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# setting this to see all columns
pd.set_option('display.max_columns', 1000)
# lets view the data
train_df.head()
# some information about the data, and the scale of the features
train_df.describe().T
# a bit more information
train_df.info()
# missing count and percentge of feature
# TODO make this a function
most_missing = train_df.isnull().sum().sort_values(ascending = False)
total = most_missing[most_missing != 0]
percent = total / len(train_df)

pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# Checking correlation of features to the target variable
(train_df.corr() ** 2)["SalePrice"].sort_values(ascending = False)
plt.subplots(figsize = (15,10))
sns.scatterplot(train_df.OverallQual, train_df.SalePrice)
plt.subplots(figsize = (15,10))
sns.scatterplot(train_df.GrLivArea, train_df.SalePrice)
# The two point to the bottom right are obvious outliers, so it will be good to remove them
train_df = train_df.drop(train_df[train_df['GrLivArea']>4500].index).reset_index(drop=True)
plt.subplots(figsize = (15,10))
sns.scatterplot(train_df.GarageCars, train_df.SalePrice)
plt.subplots(figsize = (15,10))
sns.scatterplot(train_df.GarageArea, train_df.SalePrice)
plt.subplots(figsize = (15,10))
sns.scatterplot(train_df.TotalBsmtSF, train_df.SalePrice)
# Now lets check the skew of the target variable

import matplotlib.gridspec as gridspec
from scipy import stats

fig = plt.figure(constrained_layout=True, figsize=(10,5))
grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = fig.add_subplot(grid[0, 0])
ax1.set_title('Histogram')
sns.distplot(train_df.SalePrice, norm_hist=True, ax = ax1)
 
ax2 = fig.add_subplot(grid[0, 1])
ax2.set_title('Prop_plot') 
stats.probplot(train_df.SalePrice, plot = ax2)
# Since it is right skewed we will use log transform to normalize it
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

fig = plt.figure(constrained_layout=True, figsize=(15,5))
grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = fig.add_subplot(grid[0, 0])
ax1.set_title('Histogram')
sns.distplot(train_df.SalePrice, norm_hist=True, ax = ax1)
 
ax2 = fig.add_subplot(grid[0, 1])
ax2.set_title('Prop_plot') 
stats.probplot(train_df.SalePrice, plot = ax2)
# Now lets check the correlation between our independent variables

plt.subplots(figsize = (30,30))
sns.heatmap(train_df.corr(), 
            annot=True, 
            center = 0, 
           );
# ok, now since we did all of this we will merge all the data together to transform it
# this is not good for real-world applications, since it wont give the most accurate result, since it causes data leaking
# but this is a competition and we have all the data, so we can ignore that, since this way we will get better results

all_df = pd.concat((train_df, test_df)).reset_index(drop = True)
all_df.drop(['SalePrice', 'Id'], axis = 1, inplace = True)
all_df.head()
# How many missing values

most_missing = all_df.isnull().sum().sort_values(ascending = False)
total = most_missing[most_missing != 0]
percent = total / len(all_df)

pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# This cell will impute None, where the missing data actualy has a meaning
fill_with_none = ['Alley', 
                  'PoolQC', 
                  'MiscFeature',
                  'Fence',
                  'FireplaceQu',
                  'GarageType',
                  'GarageFinish',
                  'GarageQual',
                  'GarageCond',
                  'BsmtQual',
                  'BsmtCond',
                  'BsmtExposure',
                  'BsmtFinType1',
                  'BsmtFinType2',
                  'MasVnrType']

for col in fill_with_none:
    all_df[col] = all_df[col].fillna('None')
# Same as the cell above, but for numeric features
fill_with_zero = ['BsmtFinSF1',
                  'BsmtFinSF2',
                  'BsmtUnfSF',
                  'TotalBsmtSF',
                  'BsmtFullBath', 
                  'BsmtHalfBath', 
                  'GarageYrBlt',
                  'GarageArea',
                  'GarageCars',
                  'MasVnrArea']

for col in fill_with_zero:
    all_df[col] = all_df[col].fillna(0)
# Imputing the most occurring value, there where the missing value is a missing value
fill_with_mode = ['MSZoning', 
                     'Functional',
                     'Utilities',
                     'Exterior1st',
                     'Exterior2nd',
                     'KitchenQual',
                     'SaleType', 
                     'Electrical',
                    ]

for col in fill_with_mode:
    all_df[col] = all_df[col].fillna(all_df[col].mode()[0])     
# Converting some numerical variables to categorical, as they should have been that from the start
convert_to_string = ['MSSubClass',
                     'OverallCond',
                     'OverallQual',
                     'YrSold', 
                     'MoSold']

for col in convert_to_string:
    all_df[col] = all_df[col].astype(str)  
# Lot frontage has a high dependency on Neighborhoods, so we impute with groupby
all_df['LotFrontage'] = all_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
# Check for missing data
most_missing = all_df.isnull().sum().sort_values(ascending = False)
total = most_missing[most_missing != 0]
percent = total / len(all_df)

pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# Now lets normalize the features
from scipy.stats import skew

numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_feats
# Normalizing if skew is greater than 0.75
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skewed_feats[abs(skewed_feats) > 0.75].index

for col in high_skew:
    all_df[col] = boxcox1p(all_df[col], boxcox_normmax(all_df[col] + 1))


# Check skew again
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_feats
# Adding some true or false features(Has no effect on normal linear regresion, but does od modified versions, and other models)
all_df['haspool'] = all_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_df['has2ndfloor'] = all_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_df['hasgarage'] = all_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_df['hasbsmt'] = all_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_df['hasfireplace'] = all_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# Adding features for total SF and porch SF
all_df['TotalSF'] = (all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF'])
all_df['Total_porch_sf'] = (all_df['OpenPorchSF'] 
                              + all_df['3SsnPorch'] 
                              + all_df['EnclosedPorch'] 
                              + all_df['ScreenPorch'] 
                              + all_df['WoodDeckSF']
                             )
# Checking if there are any caterorical features that cant help us
categorical_feats = all_df.dtypes[all_df.dtypes == "object"].index
for col in categorical_feats:
    print(col)
    print(all_df[col].value_counts())
# Removing these because they dont give much info
all_df = all_df.drop(['Street', 'Utilities', 'PoolQC',], axis=1)
# Adding dummy variables(OHE)
all_df = pd.get_dummies(all_df).reset_index(drop=True)
all_df.shape
# Splitting the data
y = train_df['SalePrice'].reset_index(drop=True)
X = all_df.iloc[:len(y), :]
X_final = all_df.iloc[len(y):, :]
X.head()
# Basic linear regression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .25)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_pred, y_test)


# Ridge

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
## Assiging different sets of alpha values to explore which can be the best fit for the model. 
alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    ridge = Ridge(alpha= i, normalize=True)
    ## fit the model. 
    ridge.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = ridge.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss


for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))


print('x'*10)
    
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
# Lasso

from sklearn.linear_model import Lasso
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    lasso = Lasso(alpha= i, normalize=True)
    ## fit the model. 
    lasso.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = lasso.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss


for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))


print('x'*10)
    
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
# Elastic Net

from sklearn.linear_model import ElasticNet
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    el_net = ElasticNet(alpha= i, normalize=True)
    ## fit the model. 
    el_net.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = el_net.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss


for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))


print('x'*10)
    
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
# Gradient boosting

from xgboost import XGBRegressor

temp_rss = {}
temp_mse = {}
for i in [1e-11, 5e-10, 1e-10, 5e-9]:
    ## Assigin each model. 
    xgb = XGBRegressor(learning_rate=0.01,n_estimators=3000,
                                     max_depth=3, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     reg_alpha=abs(i))
    ## fit the model. 
    xgb.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = xgb.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss
    print(i, sep=' ')
print()


for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))


print('x'*10)
    
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
# Blending

ridge = Ridge(alpha= 0.5, normalize=True)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

lasso = Lasso(alpha= 0.0001, normalize=True)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

elnet = ElasticNet(alpha= 0.0001, normalize=True)
elnet.fit(X_train, y_train)
y_pred_elnet = ridge.predict(X_test)

xgb = XGBRegressor(learning_rate=0.01,n_estimators=3000,
                                     max_depth=3, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     reg_alpha=1e-10)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

y_pred = (y_pred_ridge + y_pred_lasso + y_pred_elnet + y_pred_xgb) / 4
# RMSE
np.sqrt(mean_squared_error(y_test, y_pred))
# Learning on entire dataset

ridge = Ridge(alpha= 0.5, normalize=True)
ridge.fit(X, y)
y_pred_ridge = ridge.predict(X_final)

lasso = Lasso(alpha= 0.0001, normalize=True)
lasso.fit(X, y)
y_pred_lasso = lasso.predict(X_final)

elnet = ElasticNet(alpha= 0.0001, normalize=True)
elnet.fit(X, y)
y_pred_elnet = ridge.predict(X_final)

xgb = XGBRegressor(learning_rate=0.01,n_estimators=3000,
                                     max_depth=3, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     reg_alpha=1e-10)
xgb.fit(X, y)
y_pred_xgb = xgb.predict(X_final)

y_pred = (y_pred_ridge + y_pred_lasso + y_pred_elnet + y_pred_xgb) / 4
# Creating submit file

data = {'Id': test_df['Id'],
        'SalePrice': np.exp(y_pred)
}
submission_df = pd.DataFrame(data)
submission_df.to_csv('houses.csv', mode='w', index=False)