#Importing library for dataframe

import pandas as pd

import numpy as np

#Importing library for suppressing warnings

import warnings

warnings.filterwarnings('ignore')

#Importing library for data visualization

import matplotlib.pyplot as plt

import seaborn as sns

#Importing library for train-test data split

from sklearn.model_selection import train_test_split

#Importing library for rescaling the features

from sklearn.preprocessing import MinMaxScaler

# Importing library to calculate r-squared

from sklearn.metrics import r2_score

#Importing RFE and Linear Regression for building model

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

#Importing statsmodel for adding a constant

import statsmodels.api as sm

from sklearn.linear_model import Lasso,LassoCV

from sklearn.model_selection import GridSearchCV

#To remove omition of columns due to huge number of columns

pd.options.display.max_columns= 300

from pandas.api.types import is_numeric_dtype

import matplotlib

%matplotlib inline
#Reading the csv file

housing_train = pd.read_csv('../input/train.csv')

housing_test = pd.read_csv('../input/test.csv')

housing_test2 = pd.read_csv('../input/test.csv')
print('The number of rows in train dataset are ',housing_train.shape[0],' and number of columns are ',housing_train.shape[1])

print('The number of rows in test dataset are ',housing_test.shape[0],' and number of columns are ',housing_test.shape[1])
housing = pd.concat ([housing_train,housing_test],sort=False)
housing.select_dtypes(include='object').head()
housing.select_dtypes(include=['float','int']).head()
housing.select_dtypes(include='object').isnull().sum()[housing.select_dtypes(include='object').isnull().sum()>0]
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    housing_train[col]=housing_train[col].fillna('None')

    housing_test[col]=housing_test[col].fillna('None')
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    housing_train[col]=housing_train[col].fillna(housing_train[col].mode()[0])

    housing_test[col]=housing_test[col].fillna(housing_test[col].mode()[0])
housing.select_dtypes(include=['int','float']).isnull().sum()[housing.select_dtypes(include=['int','float']).isnull().sum()>0]
for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):

    housing_train[col]=housing_train[col].fillna(0)

    housing_test[col]=housing_test[col].fillna(0)
housing_train['LotFrontage']=housing_train['LotFrontage'].fillna(housing_train['LotFrontage'].mean())

housing_test['LotFrontage']=housing_test['LotFrontage'].fillna(housing_test['LotFrontage'].mean())
print(housing_train.isnull().sum().sum())

print(housing_test.isnull().sum().sum())
plt.figure(figsize=[30,15])

sns.heatmap(housing_train.corr(), annot=True)
#from 2 features high correlated, removing the less correlated with SalePrice

housing_train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)

housing_test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
len_housing_train=housing_train.shape[0]

print(housing_train.shape)
housing=pd.concat([housing_train,housing_test], sort=False)
housing['MSSubClass']=housing['MSSubClass'].astype(str)
from scipy.stats import skew

skew=housing.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skew_df=pd.DataFrame({'Skew':skew})

skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]
skewed_df.index
housing_train=housing[:len_housing_train]

housing_test=housing[len_housing_train:]
from scipy.special import boxcox1p

lam=0.1

for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',

       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',

       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',

       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',

       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',

       'GarageYrBlt'):

    housing_train[col]=boxcox1p(housing_train[col],lam)

    housing_test[col]=boxcox1p(housing_test[col],lam)
housing_train['SalePrice']=np.log(housing_train['SalePrice'])
housing=pd.concat([housing_train,housing_test], sort=False)

housing=pd.get_dummies(housing)
housing_train=housing[:len_housing_train]

housing_test=housing[len_housing_train:]
housing_train.drop('Id', axis=1, inplace=True)

housing_test.drop('Id', axis=1, inplace=True)
x=housing_train.drop('SalePrice', axis=1)

y=housing_train['SalePrice']

housing_test=housing_test.drop('SalePrice', axis=1)
from sklearn.preprocessing import StandardScaler, RobustScaler

sc=RobustScaler()

x=sc.fit_transform(x)

housing_test=sc.transform(housing_test)
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}





lasso = Lasso()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(x, y) 
cv_results = pd.DataFrame(model_cv.cv_results_)

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
alpha =0.001



lasso = Lasso(alpha=alpha)

        

lasso.fit(x,y)
pred=lasso.predict(housing_test)

preds=np.exp(pred)
output=pd.DataFrame({'Id':housing_test2.Id, 'SalePrice':preds})

output.to_csv('submission.csv', index=False)
output.head()