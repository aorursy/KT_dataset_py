# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head(5)
test.head(5)
train.info()
test.info()
train.keys()
train.isna().sum()
train.nunique()
train.corr()
import featuretools as ft
list(train.columns)
sns.distplot(train['SalePrice'], kde=False)
train.shape
train.keys()
train = train.drop(['Id'], axis = 1)
train.drop(train.head(1).index,inplace=True)
threshold = 0.9



# Absolute value correlation matrix

corr_matrix = train.corr().abs()

corr_matrix.head()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()
# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %d columns to remove.' % (len(to_drop)))
train = train.drop(columns = to_drop)
train.shape
train_missing = (train.isnull().sum() / len(train)).sort_values(ascending = False)

train_missing.head()
train_missing = train_missing.index[train_missing > 0.75]
all_missing = list(set(train_missing))
len(all_missing)
train.drop(columns = train_missing )
train = train.drop(columns = 'MiscFeature')
train = train.drop(columns = 'Alley')
train = train.drop(columns = 'Fence')
train = train.drop(columns = 'FireplaceQu')
train.head()
train.shape
X = train.iloc[:, :-1].values

y = train.iloc[:, -1].values
X.shape
y.shape
train['SalePrice']
y
import lightgbm as lgb

feature_importances = np.zeros(train.shape[1])



# Create the model with several hyperparameters

model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')
import warnings

warnings.filterwarnings('ignore')
 #for i in range(2):

    

    # Split into training and validation set

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)

    

    # Train using early stopping

   # model.fit(X_train, y_train, early_stopping_rounds=100, eval_set = [(X_test, y_test)], 

              #eval_metric = 'auc', verbose = 200)

    # Record the feature importances

    #feature_importances += model.feature_importances_
#feature_importances = feature_importances / 2

#feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)



#feature_importances.head()
#zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

#print('There are %d features with 0.0 importance' % len(zero_features))

#feature_importances.tail()
from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median')),

             ('pca', PCA())])
train.head()
train['MSZoning']
train['Street']
train['LotShape']
train['LandContour']
train['Utilities']
train['LotConfig']
train['LandSlope']
train= train.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope'], axis=1)
train.shape
train.keys()
train['Neighborhood']
train['Condition1']
train['Condition2']
train['BldgType']
train['HouseStyle']
train['RoofStyle']
train['RoofMatl']
train['Exterior1st']
train['ExterCond']
train['Foundation']
train['MasVnrType']
train['BsmtCond']
train['BsmtExposure']
train[ 'BsmtFinType1']
train[ 'BsmtFinType2']
train[ 'Heating']
train['HeatingQC']
train['CentralAir']
train[ 'Electrical']
train['KitchenQual']
train[ 'Functional']
train['GarageType' ]
train['GarageFinish']
train['GarageQual']
train['GarageCond']
train['PavedDrive']
train[ 'PoolQC']
train[ 'SaleType']
train[ 'SaleCondition']
train= train.drop(['Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','ExterCond','Foundation','MasVnrType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','SaleType','SaleCondition'], axis=1)
train.shape
train.info()
train['Exterior2nd']
train['ExterQual']
train['BsmtQual']
train = train.drop(['Exterior2nd','ExterQual','BsmtQual'],axis = 1)
train.shape
train.info
train_pca = pipeline.fit_transform(train)
pca = pipeline.named_steps['pca']
plt.figure(figsize = (10, 8))

plt.plot(list(range(train.shape[1])), np.cumsum(pca.explained_variance_ratio_), 'r-')

plt.xlabel('Number of PC'); plt.ylabel('Cumulative Variance Explained');

plt.title('Cumulative Variance Explained with PCA');
X = train.iloc[:, :-1].values

y = train.iloc[:, -1].values
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)

print("Mean Absolute Error from Imputation:")

print((imputed_X_train, imputed_X_test, y_train, y_test))
import pandas as pd



def clean_dataset(X_train):

    assert isinstance(X_train, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~X_train.isin([np.nan, np.inf, -np.inf]).any(1)

    return X_train[indices_to_keep].astype(np.float64)
def clean_dataset(y_train):

    assert isinstance(y_train, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~y_train.isin([np.nan, np.inf, -np.inf]).any(1)

    return y_train[indices_to_keep].astype(np.float64)
def clean_dataset(X_test):

    assert isinstance(X_test, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~X_test.isin([np.nan, np.inf, -np.inf]).any(1)

    return X_test[indices_to_keep].astype(np.float64)
def clean_dataset(y_test):

    assert isinstance(y_test, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~y_test.isin([np.nan, np.inf, -np.inf]).any(1)

    return y_test[indices_to_keep].astype(np.float64)
X_train = train.fillna(method='ffill').values
y_train = train.fillna(method='ffill').values
X_test = train.fillna(method='ffill').values
y_test = train.fillna(method='ffill').values
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test =  sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_X.fit_transform(y_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
X_test
y_pred = regressor.predict(X_test)

X.shape
X_test.shape
X_train.shape
y_train.shape
y.shape
y_pred.shape
y_test.shape
from sklearn.metrics import mean_squared_error, r2_score
print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred)

     )
def adj_r2(r2score, train):

    adj_r2 = (1 - (1 - r2score) * ((train.shape[0] - 1) / 

          (train.shape[0] - train.shape[1] - 1)))

    return adj_r2
test.shape
test.info()
test = test.drop(['Exterior2nd','ExterQual','BsmtQual','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','ExterCond','Foundation','MasVnrType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','SaleType','SaleCondition','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','MiscFeature','FireplaceQu','Fence'], axis=1)
test.info()
train.info()
test.shape
test['Alley']
test= test.drop(['Alley'], axis = 1)
test.shape
train.shape
train.head(10)
test.head(10)
X_test.shape
test.shape
def adj_r2(r2score, train):

    adj_r2 = (1 - (1 - r2score) * ((train.shape[0] - 1) / 

          (train.shape[0] - train.shape[1] - 1)))

    return adj_r2
adj_r2(r2_score(y_test, y_pred), X_train)
import statsmodels.formula.api as sm

X_ = np.append(arr = np.ones((1459, 1)).astype(int), values = X, axis = 1)
import statsmodels.api as sm
plt.scatter(X[:,0],y, color='red')

plt.xlabel('R&D')

plt.ylabel('Profit')

plt.show()
sns.distplot(train['SalePrice'], hist = False)
plt.scatter(X[:,1],y, color='red')

plt.xlabel('Adm')

plt.ylabel('Profit')

plt.show()
regressor.coef_
regressor.intercept_
regressor.singular_
train.shape
train.shape
test.shape
X_test.shape
X_train.shape
y
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet
model = Ridge()

model.fit(X_train, y_train) #fitting the model

    

y_train_pred = np.expm1(model.predict(y_train)) #predicting and converting back from log(SalePrice)

y_test_pred = np.expm1(model.predict(X_test))
X = train.drop(['SalePrice'], axis=1)

y = np.log(train.SalePrice)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

from sklearn import linear_model

lr = linear_model.LinearRegression()
train = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(train.isnull().sum() != 0)
sub = pd.DataFrame()

sub['Id'] = test["Id"]

sub['SalePrice'] = np.expm1(y)

sub.to_csv('submission.csv',index=False)