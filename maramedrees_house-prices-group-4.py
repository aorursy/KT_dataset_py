import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error

import sys

#!conda install --yes --prefix {sys.prefix} pandas_profiling

!{sys.executable} -m pip install pandas_profiling
from pandas_profiling import ProfileReport

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
profile_report = ProfileReport(train, title='Profile Report', html={'style':{'full_width':True}})
profile_report.to_notebook_iframe()
#Scoring Functions

def model_score(pred_y, y):

    print('R2: {}'.format(r2_score(pred_y,y)))

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(pred_y,y))))

def model_validation(model, x_train, x_test, y_train, y_test):

          #Model Parameters

          print(model)

          #Training Score

          print("Train Set")

          pred_train = model.predict(x_train)

          model_score(pred_train,y_train)

          

          print("Test Set")

          pred_test = model.predict(x_test)

          model_score(pred_test,y_test)
# Percentile of missing values

missing_values_count = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)

plt.figure(figsize=(15,10))

plt.xlabel('Features', fontsize=15)

plt.ylabel('Missing Values', fontsize=15)

plt.title('Percentile of Missing Values', fontsize=15)

sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10], color ='blue');   
def missing_percentage(df):

    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_percentage(train)
missing_percentage(test)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sns.boxplot(x=train['SalePrice'])
sns.boxplot(x=train['TotRmsAbvGrd'])
sns.boxplot(x=train['Fireplaces'])
sns.boxplot(x=train['TotalBsmtSF'])
sns.boxplot(x=train['GrLivArea'])
upper_lim = train['SalePrice'].quantile(.95)

lower_lim = train['SalePrice'].quantile(.05)

print(train.shape)

train = train[(train['SalePrice'] < upper_lim) & (train['SalePrice'] > lower_lim)]

print(train.shape)
upper_lim = train['GrLivArea'].quantile(.95)

lower_lim = train['GrLivArea'].quantile(.05)

train = train[(train['GrLivArea'] < upper_lim) & (train['GrLivArea'] > lower_lim)]
upper_lim = train['Fireplaces'].quantile(.95)

lower_lim = train['Fireplaces'].quantile(.05)

train = train[(train['Fireplaces'] < upper_lim) & (train['Fireplaces'] > lower_lim)]
upper_lim = train['TotalBsmtSF'].quantile(.95)

lower_lim = train['TotalBsmtSF'].quantile(.05)

train = train[(train['TotalBsmtSF'] < upper_lim) & (train['TotalBsmtSF'] > lower_lim)]
upper_lim = train['TotRmsAbvGrd'].quantile(.95)

lower_lim = train['TotRmsAbvGrd'].quantile(.05)

train = train[(train['TotRmsAbvGrd'] < upper_lim) & (train['TotRmsAbvGrd'] > lower_lim)]
# Ensuring outliers were removed using a scatter plot

GrLivArea = train['GrLivArea']

SalePrice = train['SalePrice']

plt.scatter(GrLivArea, SalePrice, edgecolors='r')

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()
# Replacing missing values with'None' in the train dataset

missing_cat_col = ["Alley", 

                   "PoolQC", 

                   "MiscFeature",

                   "Fence",

                   "FireplaceQu",

                   "GarageType",

                   "GarageFinish",

                   "GarageQual",

                   "GarageCond",

                   'BsmtQual',

                   'BsmtCond',

                   'BsmtExposure',

                   'BsmtFinType1',

                   'BsmtFinType2',

                   'MasVnrType']



for i in missing_cat_col:

    train[i] = train[i].fillna('None')
# Replacing missing values with'None' in the test dataset

missing_num_col = ['BsmtFinSF1',

                    'BsmtFinSF2',

                    'BsmtUnfSF',

                    'TotalBsmtSF',

                    'BsmtFullBath', 

                    'BsmtHalfBath', 

                    'GarageYrBlt',

                    'GarageArea',

                    'GarageCars',

                    'MasVnrArea']



for i in missing_num_col:

    train[i] = train[i].fillna(0)

    

## Replacing missing values in 'LotFrontage' by imputing the mean value of each neighborhood 

train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))
## Converting 'MSSubClass' from numerical to categorical variable

train['MSSubClass'] = train['MSSubClass'].astype(str)

train['MSZoning'] = train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
## Converting 'YrSold' and 'MoSold' from numerical to categorical variables

train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)
train['Functional'] = train['Functional'].fillna('Typ') 

train['Utilities'] = train['Utilities'].fillna('AllPub') 

train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0]) 

train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])

train['KitchenQual'] = train['KitchenQual'].fillna("TA") 

train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])

train['Electrical'] = train['Electrical'].fillna("SBrkr") 
for i in missing_cat_col:

    test[i] = test[i].fillna('None')
## Zero was used to replace the missing values 

for i in missing_num_col:

    test[i] = test[i].fillna(0)

    

test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))
test['MSSubClass'] = test['MSSubClass'].astype(str)

test['MSZoning'] = test.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test['YrSold'] = test['YrSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str) 
test['Functional'] = test['Functional'].fillna('Typ') 

test['Utilities'] = test['Utilities'].fillna('AllPub') 

test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0]) 

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna("TA") 

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

test['Electrical'] = test['Electrical'].fillna("SBrkr") 
# Removing imbalanced features to avoid overfitting model

# Removing columns with over 95% missing values

drop_col = ['3SsnPorch','BsmtHalfBath','Heating','KitchenAbvGr','LandSlope','LotArea','LowQualFinSF',

            'RoofMatl','ScreenPorch','Street','Utilities', 'MiscVal', 'PoolArea', 'MiscFeature','RoofStyle','SaleCondition','SaleType']

train.drop(columns=drop_col,axis=1, inplace=True)

test.drop(columns=drop_col,axis=1, inplace=True)

train.columns
## Dropping the 'Id' from train and test datasets 

test_id = test['Id']



train.drop(columns=['Id'],axis=1, inplace=True)

test.drop(columns=['Id'],axis=1, inplace=True)



y = train['SalePrice'].reset_index(drop=True)



train.drop(columns=['SalePrice'],axis=1, inplace=True)
# Combining datasets

dataframes = [train, test]

con_datasets = pd.concat(dataframes)
con_datasets_d = pd.get_dummies(con_datasets, drop_first=True)
train_cleaned = con_datasets_d.iloc[:len(y), :] # Train

test_cleaned = con_datasets_d.iloc[len(y):, :]  # Test
train_cleaned.shape
test_cleaned.shape
#scaling the features

from sklearn.preprocessing import StandardScaler

ss = StandardScaler().fit(train_cleaned)

train_ss = ss.transform(train_cleaned)

test_ss = ss.transform(test_cleaned)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_ss, y,test_size = .3, random_state = 0)
from sklearn.model_selection import GridSearchCV 

from sklearn.svm import SVR

gsc = GridSearchCV(

        estimator=SVR(kernel='rbf'),

        param_grid={

            'C': [0.1, 1, 100, 1000],

            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],

            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

        },

        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
from sklearn.linear_model import Lasso ,ElasticNet, BayesianRidge

els = ElasticNet()

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

ratio = [0.001,0.1,0.4,0.5]

param_grid = dict(alpha=alpha , l1_ratio=ratio)

grid_els_model = GridSearchCV(estimator=els, param_grid=param_grid, scoring='r2', verbose=1,n_jobs = -1,pre_dispatch='2*n_jobs',cv =5)

grid_els = grid_els_model.fit(X_train, y_train)

model_validation(grid_els,X_train, X_test, y_train, y_test)
lasso = Lasso()

parameters = {'alpha':[1e-03,0.01,0.1,0.5,0.8,1], 'normalize':[True,False], 'tol':[1e-06,1e-05,5e-05,1e-04,5e-04,1e-03]}

grid_lasso_model = GridSearchCV(lasso, parameters, cv=10, verbose=1, scoring = 'r2',n_jobs = -1, pre_dispatch='2*n_jobs')

grid_lasso = grid_lasso_model.fit(X_train, y_train)

model_validation(grid_lasso,X_train, X_test, y_train, y_test)
br = BayesianRidge()

alpha = [1e-15,1e-10,1e-8,1e-4,1e-2]

#ratio = [0.001,0.1,0.4,0.5]

param_grid = {'alpha_1':alpha, 'alpha_2':alpha}

grid_br_model = GridSearchCV(estimator=br,param_grid=param_grid,scoring='r2', verbose=1,n_jobs = -1,pre_dispatch='2*n_jobs',cv =5)

grid_br = grid_br_model.fit(X_train, y_train)

model_validation(grid_br,X_train, X_test, y_train, y_test)
pred_y = grid_br.predict(test_ss)

pred_y
pred = pd.DataFrame()

pred['Id'] = test_id

pred['SalePrice'] = pred_y

pred.to_csv('14th_model.csv',index=False)