#importing all the required libraries:



import numpy as np

import pandas as pd



#for visualization:

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



#import pandas_profiling package for a quick overview of the dataset (Please install this package)

import pandas_profiling as pp



#to scale data:

from sklearn.preprocessing import scale



#for building model:

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score





#to display all rows and columns:

pd.set_option('display.max_rows',500)

pd.set_option('display.max_columns',500)

pd.set_option('display.width',500)
#loading the data



df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

tes=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df.head()
tes.head()
df.shape

tes.shape
#pp.ProfileReport(df)
round(100*(tes.isnull().sum()/len(tes.index)),2)
def c(A):

    tes[A]=tes[A].replace(0,np.nan)

    tes[A].fillna((tes[A].median()),inplace=True)

    return
c('LotFrontage')

c('MasVnrArea')

c('TotalBsmtSF')

c('BsmtFinSF1')

c('BsmtFinSF2')

c('BsmtHalfBath')

c('BsmtFullBath')

c('GarageYrBlt')

c('GarageArea')

c('GarageCars')

c('BsmtUnfSF')
tes['MSZoning'].describe()
sns.countplot(tes['MSZoning'])
tes['MSZoning']=tes['MSZoning'].replace(np.nan,'RL')
sns.countplot(tes['Alley'])
tes['Alley']=tes['Alley'].replace(np.nan,'No alley access')
sns.countplot(tes['Utilities'])
tes['Utilities']=tes['Utilities'].replace(np.nan,'AllPub')
sns.countplot(tes['Exterior1st'])
tes['Exterior1st']=tes['Exterior1st'].replace(np.nan,'Other')
sns.countplot(tes['Exterior2nd'])
tes['Exterior2nd']=tes['Exterior2nd'].replace(np.nan,'Other')
tes['MasVnrType'].describe()
sns.countplot(tes['MasVnrType'])
tes['MasVnrType']=tes['MasVnrType'].replace(np.nan,'None')
sns.countplot(tes['BsmtQual'])
tes['BsmtQual']=tes['BsmtQual'].replace(np.nan,'NA')
sns.countplot(tes['BsmtCond'])
tes['BsmtCond']=tes['BsmtCond'].replace(np.nan,'NA')
tes['BsmtExposure']=tes['BsmtExposure'].replace(np.nan,'NA')
tes['BsmtFinType1']=tes['BsmtFinType1'].replace(np.nan,'NA')
tes['BsmtFinType2']=tes['BsmtFinType2'].replace(np.nan,'NA')
sns.countplot(tes['KitchenQual'])
tes['KitchenQual']=tes['KitchenQual'].replace(np.nan,'TA')
sns.countplot(tes['Functional'])
tes['Functional']=tes['Functional'].replace(np.nan,'Typ')
sns.countplot(tes['FireplaceQu'])
tes['FireplaceQu']=tes['FireplaceQu'].replace(np.nan,'NA')
tes['GarageType']=tes['GarageType'].replace(np.nan,'NA')
tes['GarageFinish']=tes['GarageFinish'].replace(np.nan,'NA')
tes['GarageQual']=tes['GarageQual'].replace(np.nan,'NA')
tes['GarageCond']=tes['GarageCond'].replace(np.nan,'NA')
tes['PoolQC']=tes['PoolQC'].replace(np.nan,'NA')
tes['Fence']=tes['Fence'].replace(np.nan,'NA')
tes['MiscFeature']=tes['MiscFeature'].replace(np.nan,'NA')
sns.countplot(tes['SaleType'])
tes['SaleType']=tes['SaleType'].replace(np.nan,'Oth')
def col(A):

    df[A]=df[A].replace(0,np.nan)

    df[A].fillna((df[A].median()),inplace=True)

    return
col('2ndFlrSF')

col('3SsnPorch')

col('BsmtFinSF1')

col('BsmtFinSF2')

col('BsmtFullBath')

col('BsmtHalfBath')

col('BsmtUnfSF')

col('EnclosedPorch')

col('Fireplaces')

col('GarageArea')

col('GarageCars')

col('HalfBath')

col('LowQualFinSF')

col('MasVnrArea')

col('MiscVal')

col('OpenPorchSF')

col('PoolArea')

col('ScreenPorch')

col('TotalBsmtSF')

col('WoodDeckSF')
#1. 'Alley':



df['Alley'].describe()
sns.countplot(df['Alley'])
df['Alley']=df['Alley'].replace(np.nan,'No alley access')
sns.countplot(df['Alley'])
#2. BsmtCond:



df['BsmtCond'].describe()
sns.countplot(df['BsmtCond'])
df['BsmtCond']=df['BsmtCond'].replace(np.nan,'NA')
sns.countplot(df['BsmtCond'])
#3.BsmtExposure:



df['BsmtExposure'].describe()
sns.countplot(df['BsmtExposure'])
df['BsmtExposure']=df['BsmtExposure'].replace(np.nan,'NA')
sns.countplot(df['BsmtExposure'])
#4. BsmtFinType1:



df['BsmtFinType1'].describe()
sns.countplot(df['BsmtFinType1'])
df['BsmtFinType1']=df['BsmtFinType1'].replace(np.nan,'NA')
sns.countplot(df['BsmtFinType1'])
#5.BsmtFinType2:



df['BsmtFinType2'].describe()
sns.countplot(df['BsmtFinType2'])
df['BsmtFinType2']=df['BsmtFinType2'].replace(np.nan,'NA')
sns.countplot(df['BsmtFinType2'])
#6. BsmtQual:



df['BsmtQual'].describe()
sns.countplot(df['BsmtQual'])
df['BsmtQual']=df['BsmtQual'].replace(np.nan,'NA')
sns.countplot(df['BsmtQual'])
#7.Fence

df['Fence'].describe()
sns.countplot(df['Fence'])
df['Fence']=df['Fence'].replace(np.nan,'NA')
sns.countplot(df['Fence'])
#8.FireplaceQu

df['FireplaceQu'].describe()
sns.countplot(df['FireplaceQu'])
df['FireplaceQu']=df['FireplaceQu'].replace(np.nan,'NA')
sns.countplot(df['FireplaceQu'])
#9.GarageCond



df['GarageCond'].describe()
sns.countplot(df['GarageCond'])
df['GarageCond']=df['GarageCond'].replace(np.nan,'NA')
sns.countplot(df['GarageCond'])
#10. GarageFinish



df['GarageFinish'].describe()
sns.countplot(df['GarageFinish'])
df['GarageFinish']=df['GarageFinish'].replace(np.nan,'NA')
sns.countplot(df['GarageFinish'])
#11.GarageQual



df['GarageQual'].describe()
sns.countplot(df['GarageQual'])
df['GarageQual']=df['GarageQual'].replace(np.nan,'NA')
sns.countplot(df['GarageQual'])
#12.GarageType



df['GarageType'].describe()
sns.countplot(df['GarageType'])
df['GarageType']=df['GarageType'].replace(np.nan,'NA')
sns.countplot(df['GarageType'])
#13.GarageYrBlt 



df['GarageYrBlt'].describe()
sns.boxplot(df['GarageYrBlt'])
df['GarageYrBlt'].fillna((df['GarageYrBlt'].mean()), inplace=True)

df['GarageYrBlt'].describe()
sns.boxplot(df['GarageYrBlt'])
#14. LotFrontage:



df['LotFrontage'].describe()
sns.boxplot(df['LotFrontage'])
df['LotFrontage'].fillna((df['LotFrontage'].median()),inplace=True)
df['LotFrontage'].describe()
#15.MiscFeature:



df['MiscFeature'].describe()
sns.countplot(df['MiscFeature'])
df['MiscFeature']=df['MiscFeature'].replace(np.nan,'NA')
sns.countplot(df['MiscFeature'])
#16.PoolQC:



df['PoolQC'].describe()
sns.countplot(df['PoolQC'])
df['PoolQC']=df['PoolQC'].replace(np.nan,'NA')
sns.countplot(df['PoolQC'])
#17.Electrical:



df['Electrical'].describe()
sns.countplot(df['Electrical'])
df['Electrical']=df['Electrical'].replace(np.nan,'SBrkr')
#18. MasVnrType:



df['MasVnrType'].describe()
sns.countplot(df['MasVnrType'])
df['MasVnrType']=df['MasVnrType'].replace(np.nan,'None')
round(100*(df.isnull().sum()/len(df.index)),2)
#Visualizing categorical data



plt.figure(figsize=(100, 50))

plt.subplot(3,3,1)

sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data = df)

plt.subplot(3,3,2)

sns.boxplot(x = 'GarageType', y = 'SalePrice', data = df)

plt.subplot(3,3,3)

sns.boxplot(x = 'GarageFinish' , y = 'SalePrice', data = df)

plt.subplot(3,3,4)

sns.boxplot(x = 'Fence', y = 'SalePrice', data = df)

plt.subplot(3,3,5)

sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = df)

plt.subplot(3,3,6)

sns.boxplot(x = 'BsmtFinType2', y = 'SalePrice', data = df)

plt.subplot(3,3,7)

sns.boxplot(x=  'Electrical', y ='SalePrice', data = df)

plt.subplot(3,3,8)

sns.boxplot(x=  'CentralAir',y='SalePrice', data = df)

plt.show()
#df['YearBuilt']
#df['Houseage']=2019-df['YearBuilt']

#tes['Houseage']=2019-tes['YearBuilt']
#df['Houseage']
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = df.drop(['Id','SalePrice'], axis=1)

X_test=tes.drop(['Id'], axis=1)

X.head()
X.shape
X_test.shape
#Putting response variable to y

y =df['SalePrice']



y.head()
#creating dummy variables for categorical variables



# selecting all categorical variables

df_cat = X.select_dtypes(include=['object'])

tes_cat = X_test.select_dtypes(include=['object'])

df_cat.head()
# convert categorical variables into dummies

df_dummies = pd.get_dummies(df_cat, drop_first=True)

df_dummies.head()
tes_dummies = pd.get_dummies(tes_cat, drop_first=True)

tes_dummies.head()
test_dummies = tes_dummies.reindex(columns = df_dummies.columns,fill_value=0)
# drop categorical variables 

X = X.drop(list(df_cat.columns), axis=1)
X_test = X_test.drop(list(tes_cat.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, df_dummies], axis=1)

X_test = pd.concat([X_test, test_dummies], axis=1)
X_test.shape
X.shape
from sklearn.preprocessing import StandardScaler



# storing column names in cols,

cols = X.columns

X = pd.DataFrame(StandardScaler().fit_transform(X))

X.columns = cols

X.columns

# storing column names in cols,

cs = X_test.columns

X_test = pd.DataFrame(StandardScaler().fit_transform(X_test))

X_test.columns = cs

X_test.columns
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}

# cross validation

folds = 5



lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X, y) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plotting mean test and train scores with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv.best_params_
#alpha =500



lasso = Lasso(alpha=500)

        

lasso.fit(X, y) 
#Extracting the coefficients and model equation from lasso regression

lasso.coef_
# lasso model parameters generation

model_parameters = list(lasso.coef_)

model_parameters.insert(0, lasso.intercept_)

model_parameters = [round(x, 1) for x in model_parameters]

cols = X.columns

cols = cols.insert(0, "constant")

print(list(zip(cols, model_parameters)))
X_test.head()
# model with optimal alpha=500

lm1 = Lasso(alpha=500)

lm1.fit(X, y)



from sklearn.metrics import r2_score

# predict

y_train_pred = lm1.predict(X)
r_square_score=r2_score(y,y_train_pred)

print("R Square score:{}".format(r_square_score))
y_test_pred = lm1.predict(X_test)
y_test_pred
my_submission = pd.DataFrame({'Id': tes.Id, 'SalePrice': y_test_pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_hp_lasso8.csv', index=False)
# list of alphas to tune

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

model_cv.fit(X, y) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=140]

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv.best_params_

#alpha = 500

ridge = Ridge(alpha=500)



ridge.fit(X, y)

#Predictor Variables from the Model built using Ridge Regression:

ridge.coef_
# ridge model parameters

model_parameters = list(ridge.coef_)

model_parameters.insert(0, ridge.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = X.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))
#model with optimal alpha

lm2 = Ridge(alpha=500)

lm2.fit(X, y)



from sklearn.metrics import r2_score

y_train_pred = lm2.predict(X)

y_test_pr = lm2.predict(X_test)

y_test_pr
my_submission = pd.DataFrame({'Id': tes.Id, 'SalePrice': y_test_pr})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_ridge_hp8.csv', index=False)