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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
#importing data to python notebook
housing = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

housing_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
housing.head()
#shape of the data
housing.shape
housing_test.head()
housing_test.shape
#information of the data
housing.info()
housing.describe()
housing_test.isnull().sum()
housing.isnull().sum()
#handling na values which are not actually NA
housing['Alley'].fillna('No alley access', inplace = True)

housing['FireplaceQu'].fillna('No Fireplace', inplace = True)

housing['PoolQC'].fillna('No Pool', inplace = True)

housing['Fence'].fillna('No Fence', inplace = True)

housing['GarageFinish'].fillna('No Garage', inplace = True)

housing['GarageType'].fillna('No Garage',inplace = True)

housing['GarageCond'].fillna('No Garage',inplace = True)

housing['GarageQual'].fillna('No Garage',inplace = True)

housing['BsmtExposure'].fillna('No Basement',inplace = True)

housing['BsmtFinType2'].fillna('No Basement',inplace = True)

housing['BsmtFinType1'].fillna('No Basement',inplace = True)

housing['BsmtCond'].fillna('No Basement',inplace = True)

housing['BsmtQual'].fillna('No Basement',inplace = True)

housing['MasVnrType'].fillna('None',inplace = True) 
housing['LotFrontage'] = housing['LotFrontage'].fillna(housing['LotFrontage'].median())
housing['MasVnrArea'] = housing['MasVnrArea'].fillna(housing['MasVnrArea'].median())
housing['GarageYrBlt'] = housing['GarageYrBlt'].fillna(housing['GarageYrBlt'].median())
housing['Electrical'] = housing['Electrical'].fillna(housing['Electrical'].mode()[0])
# changing to no.of years from the action taken in an year
housing['YearBuilt']=2020-housing['YearBuilt']

housing['YearRemodAdd']=2020-housing['YearRemodAdd']

housing['GarageYrBlt']=2020-housing['GarageYrBlt']

housing['YrSold']=2020-housing['YrSold']
#handling na values which are not actually NA
    

housing_test['FireplaceQu'].fillna('No Fireplace', inplace = True)

housing_test['BsmtCond'].fillna('No Basement',inplace = True)

housing_test['Fence'].fillna('No Fence', inplace = True)

housing_test['GarageFinish'].fillna('No Garage', inplace = True)

housing_test['GarageType'].fillna('No Garage',inplace = True)

housing_test['GarageCond'].fillna('No Garage',inplace = True)

housing_test['GarageQual'].fillna('No Garage',inplace = True)

housing_test['BsmtExposure'].fillna('No Basement',inplace = True)

housing_test['BsmtFinType2'].fillna('No Basement',inplace = True)

housing_test['BsmtFinType1'].fillna('No Basement',inplace = True)

housing_test['BsmtQual'].fillna('No Basement',inplace = True)

housing_test['MasVnrType'].fillna('None',inplace = True)

housing_test['Alley'].fillna('No alley access', inplace = True)
# Filling missing Continuous variables :

housing_test['LotFrontage'] = housing_test['LotFrontage'].fillna(housing_test['LotFrontage'].median())

housing_test['MasVnrArea'] = housing_test['MasVnrArea'].fillna(housing_test['MasVnrArea'].median())

housing_test['GarageYrBlt'] = housing_test['GarageYrBlt'].fillna(housing_test['GarageYrBlt'].median())

housing_test['BsmtFinSF2'] = housing_test['BsmtFinSF2'].fillna(housing_test['BsmtFinSF2'].median())

housing_test['BsmtFullBath'] = housing_test['BsmtFullBath'].fillna(housing_test['BsmtFullBath'].median())

housing_test['TotalBsmtSF'] = housing_test['TotalBsmtSF'].fillna(housing_test['TotalBsmtSF'].median())

housing_test['GarageCars'] = housing_test['GarageCars'].fillna(housing_test['GarageCars'].median())

housing_test['GarageArea'] = housing_test['GarageArea'].fillna(housing_test['GarageArea'].median())

housing_test['BsmtUnfSF'] = housing_test['BsmtUnfSF'].fillna(housing_test['BsmtUnfSF'].median())

housing_test['BsmtFinSF1'] = housing_test['BsmtFinSF1'].fillna(housing_test['BsmtFinSF1'].median())

housing_test['Functional'] = housing_test['Functional'].fillna(housing_test['Functional'].mode()[0])

housing_test['BsmtHalfBath']=housing_test['BsmtHalfBath'].fillna(housing_test['BsmtHalfBath'].mode()[0])

housing_test['MSZoning'] = housing_test['MSZoning'].fillna(housing_test['MSZoning'].mode()[0])

housing_test['Utilities']=housing_test['Utilities'].fillna(housing_test['Utilities'].mode()[0])

housing_test['Exterior1st'] = housing_test['Exterior1st'].fillna(housing_test['Exterior1st'].mode()[0])

housing_test['KitchenQual'] = housing_test['KitchenQual'].fillna(housing_test['KitchenQual'].mode()[0])

housing_test['Exterior2nd'] = housing_test['Exterior2nd'].fillna(housing_test['Exterior2nd'].mode()[0])

housing_test['SaleType'] = housing_test['SaleType'].fillna(housing_test['SaleType'].mode()[0])

housing_test['Exterior1st']=housing_test['Exterior1st'].fillna(housing_test['Exterior1st'].mode()[0])

housing_test['Exterior2nd']=housing_test['Exterior2nd'].fillna(housing_test['Exterior2nd'].mode()[0])

housing_test['KitchenQual']=housing_test['KitchenQual'].fillna(housing_test['KitchenQual'].mode()[0])

housing_test['Functional']=housing_test['Functional'].fillna(housing_test['Functional'].mode()[0])

housing_test['SaleType']=housing_test['SaleType'].fillna(housing_test['SaleType'].mode()[0])
# changing to no.of years from the action taken in an year
housing_test['YearBuilt']=2020-housing_test['YearBuilt']

housing_test['YearRemodAdd']=2020-housing_test['YearRemodAdd']

housing_test['GarageYrBlt']=2020-housing_test['GarageYrBlt']

housing_test['YrSold']=2020-housing_test['YrSold']

housing.drop(['MiscFeature'],inplace=True,axis=1)
housing_test.drop(['MiscFeature'],inplace=True,axis=1)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
fig, axs = plt.subplots(6,5, figsize = (30,18))
plt1 = sns.countplot(housing['Alley'], ax = axs[0,0])
plt2 = sns.countplot(housing['FireplaceQu'], ax = axs[0,1])
plt3 = sns.countplot(housing['PoolQC'], ax = axs[0,2])
plt4 = sns.countplot(housing['GarageFinish'], ax = axs[0,3])
plt5 = sns.countplot(housing['GarageType'], ax = axs[0,4])
plt6 = sns.countplot(housing['GarageCond'], ax = axs[1,0])
plt7 = sns.countplot(housing['GarageQual'], ax = axs[1,1])
plt8 = sns.countplot(housing['BsmtExposure'], ax = axs[1,2])
plt9 = sns.countplot(housing['BsmtFinType2'], ax = axs[1,3])
plt10 = sns.countplot(housing['BsmtFinType1'], ax = axs[1,4])
plt11 = sns.countplot(housing['BsmtCond'], ax = axs[2,0])
plt12 = sns.countplot(housing['BsmtQual'], ax = axs[2,1])
plt13 = sns.countplot(housing['MasVnrType'], ax = axs[2,2])
plt14 = sns.countplot(housing['Street'], ax = axs[2,3])

plt15 = sns.countplot(housing['LotShape'], ax = axs[2,4])
plt16 = sns.countplot(housing['LandContour'], ax = axs[3,0])
plt17 = sns.countplot(housing['Utilities'], ax = axs[3,1])
plt18 = sns.countplot(housing['LandSlope'], ax = axs[3,2])
plt19 = sns.countplot(housing['CentralAir'], ax = axs[3,2])
plt20 = sns.countplot(housing['BsmtFullBath'], ax = axs[3,4])
plt21 = sns.countplot(housing['FullBath'], ax = axs[4,0])
plt22 = sns.countplot(housing['BsmtHalfBath'], ax = axs[4,1])
plt23 = sns.countplot(housing['HalfBath'], ax = axs[4,2])
plt24= sns.countplot(housing['KitchenAbvGr'], ax = axs[4,3])
plt25 = sns.countplot(housing['KitchenQual'], ax = axs[4,4])
plt26 = sns.countplot(housing['Fireplaces'], ax = axs[5,0])
plt27 = sns.countplot(housing['FireplaceQu'], ax = axs[5,1])

plt.tight_layout()
# Dropping skewed columns which are observed from the plots:
housing.drop(['Street','Alley','LandContour','Utilities','LandSlope','BsmtCond','CentralAir','BsmtHalfBath','KitchenAbvGr','PavedDrive','PoolQC','Id'],inplace=True,axis=1)
housing.isnull().sum()
# Dropping in test set as well
housing_test.drop(['Street','Alley','LandContour','Utilities','LandSlope','BsmtCond','CentralAir','BsmtHalfBath','KitchenAbvGr','PavedDrive','PoolQC','Id'],inplace=True,axis=1)
housing_test.isnull().sum()
housing.shape
housing_test.shape
#scatter plot GrLivArea vs SalePrice
data = pd.concat([housing['SalePrice'], housing['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
housing['GrLivArea'] = np.where(housing['GrLivArea'] >4000, 4000,housing['GrLivArea'])

housing['SalePrice'] = np.where(housing['SalePrice'] >600000, 600000,housing['SalePrice'])

data = pd.concat([housing['SalePrice'], housing['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
#scatter plot LotFrontage vs SalePrice
fig, ax = plt.subplots()
ax.scatter(x = housing['LotFrontage'], y = housing['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('LotFrontage', fontsize=13)
plt.show()
housing['LotFrontage'] = np.where(housing['LotFrontage'] >200, 200,housing['LotFrontage'])

#scatter plot LotFrontage vs SalePrice
fig, ax = plt.subplots()
ax.scatter(x = housing['LotFrontage'], y = housing['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('LotFrontage', fontsize=13)
plt.show()
#GarageCars vs SalePrice 
f, ax = plt.subplots(figsize=(5, 5))
sns.boxplot(x=housing['GarageCars'], y = housing['SalePrice'])
fig, ax = plt.subplots()
ax.scatter(x = housing['GarageArea'], y = housing['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = housing['TotalBsmtSF'], y = housing['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()
housing['TotalBsmtSF'] = np.where(housing['TotalBsmtSF'] >3200, 3200,housing['TotalBsmtSF'])
fig, ax = plt.subplots()
ax.scatter(x = housing['TotalBsmtSF'], y = housing['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()
#barplot of overallqual vs SalePrice 
sns.barplot(housing.OverallQual,housing.SalePrice)
#barplot of GarageCars vs SalePrice
sns.barplot(housing.GarageCars,housing.SalePrice)
#barplot of GarageCars vs GarageArea
sns.barplot(housing.GarageCars,housing.GarageArea)
#year of sold vs SalePrice
sns.boxplot(x="YrSold",y="SalePrice",data=housing);
plt.figure(figsize=(15,5))
sns.boxplot(x="YearRemodAdd",y="SalePrice",data=housing, notch = True)
plt.figure(figsize=(35,7))
sns.boxplot(x="YearBuilt",y="SalePrice",data=housing,notch = True)
#check SalePrice distribution
plt.figure(figsize=(10,5));
plt.xlabel('xlabel', fontsize=16);
plt.rc('xtick', labelsize=14); 
plt.rc('ytick', labelsize=14); 


sns.distplot(housing['SalePrice']);
print("Skewness: %f" % housing['SalePrice'].skew())
from scipy import stats
fig = plt.figure()
res = stats.probplot(housing['SalePrice'], plot=plt)
plt.show()
#check SalePrice distribution after log transformation
plt.figure(figsize=(10,5));
plt.xlabel('xlabel', fontsize=16);
plt.rc('xtick', labelsize=14); 
plt.rc('ytick', labelsize=14); 

housing["SalePrice"] = np.log1p(housing["SalePrice"])
sns.distplot(housing.SalePrice)
fig = plt.figure()

#probability plot for saleprice
from scipy import stats
fig = plt.figure()
res = stats.probplot(np.log1p(housing['SalePrice']), plot=plt)
plt.show()
#correlation matrix
housing.corr()
#correlation plot
corrmat = housing.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);
# features which are highly correlated to the SalePrice
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(housing[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#scatterplot of highcorrelated features
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(housing[cols], size = 2.5)
plt.show();
#concating the test and train datasets
df=pd.concat([housing,housing_test],axis=0)
df = df.reset_index(drop = True)
df.shape
#df
dummy = pd.get_dummies(df[['MSZoning','LotShape','LotConfig','Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'Fence', 'SaleType', 'SaleCondition']],drop_first=True)
df = pd.concat([df,dummy],axis=1)
df.shape
df = df.drop(['MSZoning','LotShape','LotConfig','Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'Fence', 'SaleType', 'SaleCondition'],axis = 1)
df.shape
df_final=df.iloc[:1460,:]
df_Test=df.iloc[1460:,:]
df_Test.drop(['SalePrice'],axis=1,inplace=True)
#splitting the main data frame to X(independent variables) and y(target variable) 
X = df_final.drop(['SalePrice'], axis=1)
y = df_final['SalePrice']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 

#checking the value of optimum number of parameters
print(model_cv.best_params_)
print(model_cv.best_score_)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()
# plotting mean test and train scores with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
alpha = 100
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_
#predict:
from sklearn.metrics import r2_score
y_train_pred = ridge.predict(X_train)
print('R2 score of Training Data:',r2_score(y_true=y_train,y_pred=y_train_pred) )
y_test_pred = ridge.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_true=y_test,y_pred=y_test_pred))
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_test_pred))
print(rms)
lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()
#checking the value of optimum number of parameters
print(model_cv.best_params_)
print(model_cv.best_score_)
alpha =0.0001

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 
lasso.coef_
from sklearn.metrics import r2_score
y_train_pred = lasso.predict(X_train)
print('R2 score of Training Data:',r2_score(y_true=y_train,y_pred=y_train_pred) )
y_test_pred = lasso.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_true=y_test,y_pred=y_test_pred) )
rms = sqrt(mean_squared_error(y_test, y_test_pred))
print(rms)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
# random forest - the class weight is used to handle class imbalance - it adjusts the cost function
forest = RandomForestRegressor( n_jobs = -1)

# hyperparameter space
params = {"criterion": ['mse','mae'], "max_features": ['auto', 'sqrt', 'log2']}

# create gridsearch object
model = GridSearchCV(estimator=forest,param_grid=params, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)
# print best hyperparameters
from sklearn.model_selection import GridSearchCV
print("Best hyperparameters: ", model.best_params_)
# run a random forest model on train data
rf_model = RandomForestRegressor(n_estimators=100, max_features='auto',criterion = 'mse', random_state=4, verbose=1)
rf_model.fit(X_train,y_train)
y_train_pred = rf_model.predict(X_train)
print('R2 score of Training Data:',r2_score(y_true=y_train,y_pred=y_train_pred) )
y_test_pred = rf_model.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_true=y_test,y_pred=y_test_pred))
rms = sqrt(mean_squared_error(y_test, y_test_pred))
print(rms)
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train, verbose=False)
y_train_pred = my_model.predict(X_train)
print('R2 score of Training Data:',r2_score(y_true=y_train,y_pred=y_train_pred) )
y_test_pred = my_model.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_true=y_test,y_pred=y_test_pred))
rms = sqrt(mean_squared_error(y_test, y_test_pred))
print(rms)
#found ridge regression is performing well
y_pred = ridge.predict(df_Test)
##Create Sample Submission file and Submit using ANN
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)
sub_df.head()