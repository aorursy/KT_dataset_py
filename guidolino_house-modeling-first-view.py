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
import math

import seaborn as sns

import plotly.express as px

import matplotlib

import time

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')

import statsmodels.api as sm

matplotlib.rcParams['axes.labelsize'] = 20

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
from scipy import stats

from scipy.stats import norm

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 

from sklearn.model_selection import GridSearchCV
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
print(f'São {train.shape[0]} observações para {train.shape[1]} variáveis de treino')
train['flag'] = 'train'

test ['flag'] = 'test'
df = pd.concat([train,test])
df.loc[df['flag'] == 'train'].info()
df.loc[df['flag'] == 'train'].head()
plt.figure(figsize=(10,8))

sns.distplot(df.loc[df['flag'] == 'train'].SalePrice, fit=norm)

fig = plt.figure()

res = stats.probplot(df.loc[df['flag'] == 'train'].SalePrice, plot=plt)
plt.figure(figsize=(10,8))

sns.distplot(np.log(df.loc[df['flag'] == 'train'].SalePrice), fit=norm)

fig = plt.figure()

res = stats.probplot(np.log(df.loc[df['flag'] == 'train'].SalePrice), plot=plt)
df.loc[df['flag'] == 'train'].MSZoning.value_counts(dropna=False)
def plot_box (df,hue,variavel):    

    plt.figure(figsize=(10,8))

    sns.boxplot(x=variavel, y=hue, data=df)
def plot_scatter(df,hue,variavel):    

    plt.figure(figsize=(10,8))

    sns.scatterplot(x=variavel, y=hue, data=df)
plot_box(df.loc[df['flag'] == 'train'],"SalePrice","MSZoning")
plot_scatter(df.loc[df['flag'] == 'train'],"SalePrice","LotArea")
plot_box(df.loc[df['flag'] == 'train'],"SalePrice","HouseStyle")
plot_scatter(df.loc[df['flag'] == 'train'],"SalePrice","FullBath")
df.loc[df['flag'] == 'train'].OverallQual.value_counts()
plot_box(df.loc[df['flag'] == 'train'],"SalePrice","OverallQual")
plot_scatter(df.loc[df['flag'] == 'train'],"SalePrice",'YearBuilt')
df.columns
df_clean = df.drop(columns=['Alley','GarageYrBlt','GarageCars',"FireplaceQu",'KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF',"Alley","PoolQC","Fence","MiscFeature" ]).copy()
plt.figure(figsize=(25,20))

sns.heatmap(df_clean.loc[df_clean['flag'] == 'train'].corr(), annot=True)

col_cat = ['MSZoning',

'Street',

'LotShape',

'LandContour',

'Utilities',

'LotConfig',

'LandSlope',

'Neighborhood',

'Condition1',

'Condition2',

'BldgType',

'HouseStyle',

'RoofStyle',

'RoofMatl',

'Exterior1st',

'Exterior2nd',

'MasVnrType',

'ExterQual',

'ExterCond',

'Foundation',

'BsmtQual',

'BsmtCond',

'BsmtExposure',

'BsmtFinType1',

'BsmtFinType2',

'Heating',

'HeatingQC',

'CentralAir',

'Electrical',

'KitchenQual',

'Functional',

'GarageType',

'GarageFinish',

'GarageQual',

'GarageCond',

'PavedDrive',

'SaleType',

'SaleCondition',

'SalePrice']

df_clean[col_cat]
for col in col_cat:

    plot_box(df_clean[col_cat],"SalePrice",col)
#col_drop = ["Street","LotShape","LandContour","LotConfig","LandSlope","BldgType",

            #"RoofStyle","ExterCond","BsmtFinType1","BsmtFinType2","Functional","GarageCond","PavedDrive","log_sale"]
df_clean.loc[df_clean['flag'] == 'train'].info()
#df_clean.drop(columns=col_drop, inplace=True)
col_dummie = ['MSZoning',

'Street',

'LotShape',

'LandContour',

'Utilities',

'LotConfig',

'LandSlope',

'Neighborhood',

'Condition1',

'Condition2',

'BldgType',

'HouseStyle',

'RoofStyle',

'RoofMatl',

'Exterior1st',

'Exterior2nd',

'MasVnrType',

'ExterQual',

'ExterCond',

'Foundation',

'BsmtQual',

'BsmtCond',

'BsmtExposure',

'BsmtFinType1',

'BsmtFinType2',

'Heating',

'HeatingQC',

'CentralAir',

'Electrical',

'KitchenQual',

'Functional',

'GarageType',

'GarageFinish',

'GarageQual',

'GarageCond',

'PavedDrive',

'SaleType',

'SaleCondition']
df_dummies = pd.get_dummies(df_clean[col_dummie])
df_clean.drop(columns=col_dummie, inplace=True)
df_model = pd.concat([df_clean,df_dummies],axis=1)
df_model.loc[df_model['flag']=="train"].head()
def preenchimento(data):

    for col in data.columns:

        data[col] = data[col].fillna(data[col].median())

    return data
train = df_model.loc[df_model['flag']=="train"].drop(columns=(['flag'])).copy()

test = df_model.loc[df_model['flag']=="test"].drop(columns=(['flag'])).copy()
preenchimento(train)

preenchimento(test)
train.describe()
X = train.drop(['SalePrice','Id'], axis=1).copy()

y = np.log(train.SalePrice)

X_submit = test.drop(['SalePrice','Id'], axis=1).copy()

y_submit = test.Id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
SEED = 1988

np.random.seed(SEED)

ln = LinearRegression(fit_intercept = False, normalize = False)

rf = RandomForestRegressor(max_depth = 40, max_features =  50, min_samples_leaf = 2 ,n_estimators = 2400)

gbm = GradientBoostingRegressor(max_depth = 5, max_features = 32, min_samples_leaf = 2, n_estimators = 2400)

np.random.seed(SEED)



ln.fit(X_train, y_train)

ln_y_pred = ln.predict(X_test)

ln_mse = mean_squared_error((y_test), (ln_y_pred))

ln_erro = math.sqrt(ln_mse)

r2_ln_pred = r2_score(y_test, ln_y_pred)

r2_ln_log = r2_score(np.log(y_test), np.log(ln_y_pred))



rf.fit(X_train, y_train)

rf_y_pred = rf.predict(X_test)

rf_mse = mean_squared_error(np.log(y_test), np.log(rf_y_pred))

rf_erro = math.sqrt(rf_mse)

r2_rf_pred = r2_score(y_test, rf_y_pred)

r2_rf_log = r2_score(np.log(y_test), np.log(rf_y_pred))                     



gbm.fit(X_train, y_train)

gbm_y_pred = gbm.predict(X_test)

gbm_mse = mean_squared_error(np.log(y_test), np.log(gbm_y_pred))

gbm_erro = math.sqrt(gbm_mse)

r2_gbm_pred = r2_score(y_test, gbm_y_pred)

r2_gbm_log = r2_score(np.log(y_test), np.log(gbm_y_pred))



print("LinearRegression")

print("------------------------------")

print(f'MSE = {ln_erro} and R-square = {r2_ln_pred}, R-slog {r2_ln_log}')

print("------------------------------")

print("RandomForest")

print("------------------------------")

print(f'MSE = {rf_erro} and R-square = {r2_rf_pred}, R-slog {r2_rf_log}')

print("------------------------------")

print("GradientBoosting")

print("------------------------------")

print(f'MSE = {gbm_erro} and R-square = {r2_gbm_pred}, R-slog {r2_gbm_log}')

print("------------------------------")
plt.figure(figsize = (20,12))

#plt.scatter(y_test,ln_y_pred,label='LR',marker = 'o',color='r')

plt.scatter(y_test,rf_y_pred,label='RF',marker = 'o',color='b')

plt.scatter(y_test,gbm_y_pred,label='GBR',marker = 'o',color='y')

plt.title('Modelos',fontsize = 25)

plt.legend(fontsize = 20)

plt.show()
plt.figure(figsize = (20,12))

#plt.scatter(np.log(y_test),np.log(ln_y_pred),label='LR',marker = 'o',color='r')

plt.scatter(np.log(y_test),np.log(rf_y_pred),label='RF',marker = 'o',color='b')

plt.scatter(np.log(y_test),np.log(gbm_y_pred),label='GBR',marker = 'o',color='y')

plt.title('Modelos',fontsize = 25)

plt.legend(fontsize = 20)

plt.show()
plt.figure(figsize = (20,12))

#sns.distplot(np.log(ln_y_pred),color='r')

sns.distplot(np.log(rf_y_pred),color='b')

sns.distplot(np.log(gbm_y_pred),color='y')



plt.show()
y_resp = gbm.predict(X_submit)

submission = pd.concat([pd.DataFrame(y_submit),pd.DataFrame(np.exp(y_resp))], axis=1)

submission.rename(columns={0:'SalePrice'},inplace=True)

submission= submission.set_index("Id")

submission
plt.figure(figsize=(10,8))

sns.distplot(submission.SalePrice, fit=norm)

fig = plt.figure()

res = stats.probplot(submission.SalePrice, plot=plt)
plt.figure(figsize=(10,8))

sns.distplot(np.log(submission.SalePrice), fit=norm)

fig = plt.figure()

res = stats.probplot(np.log(submission.SalePrice), plot=plt)
submission.to_csv("submission.csv")