# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
test.head()
train.info()
test.info()
train.describe
test.describe
train.shape
#Feature types

train.select_dtypes(include=['int64','float64']).columns
train.select_dtypes(include=['object']).columns
train.dtypes.head()
train.isnull().sum().sort_values(ascending = False).head(20)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train['SalePrice'].count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
missing_column_train= missing_data[missing_data['Total']>0].index

missing_column_train
#Check for unique values

train['PoolQC'].unique()
train['MiscFeature'].unique()
train['Alley'].unique()
train['Fence'].unique()
train['FireplaceQu'].unique()
for i in missing_data[missing_data['Total']>0].index:

    if (train[i].dtype == 'object'):

        print(' Unique values in {} feature are {}'.format(i,train[i].unique()))
for i in missing_data[missing_data['Total']>0].index:

    if (train[i].dtype != 'object' and len(train[i].unique())<=10):

        print(' Unique values in {} feature are {}'.format(i,train[i].unique()))

#Drop the column ID 

train_ = train.drop(['Id'],axis=1)

test_ = test.drop(['Id'],axis=1)
#How to fill miss values

def missing_values(df,df2):

    

    for column in df2:

        if (df[column].isnull().sum()) >0:

            print(column +" has missing values type : "+ str(df[column].dtype))

            if df[column].dtype in ('int64','float64'):

                df[column] = df[column].fillna(df[column].mean())

            else:

                if column in ['Elecrical','MasVnrType']:

                    df[column] = df[column].fillna(df[column].dropna().mode()[0])

                else:

                    df[column] = df[column].fillna('NAN')
#Fill the missing values for train dataset

missing_values(train,missing_column_train)
missing_values(test,missing_column_train)
#In test dataset may still missing values . Lets check it

missing_test=test_.isnull().sum().sort_values(ascending=False)

test_missing_data = pd.concat([missing_test], axis=1, keys=['Total_missing'])

test_missing_data.head(20)


for i in test_missing_data[test_missing_data['Total_missing']>0].index:

    if (test[i].dtype != 'object' and len(test[i].unique())<=10):

        print(' Unique values in {} feature are {}'.format(i,test[i].unique()))


for i in test_missing_data[test_missing_data['Total_missing']>0].index:

    if (test[i].dtype == 'object'):

        print(' Unique values in {} feature are {} \n'.format(i,test[i].unique()))
def missing_values_test(df):

    

    for column in df:

        if (df[column].isnull().sum()) >0:

            print(column +" has missing values type : "+ str(df[column].dtype))

            if ((df[column].dtype in ('int64','float64')) and (column not in['BsmtFullBath','BsmtHalfBath','GarageCars'])):

                df[column] = df[column].fillna(df[column].mean())

            else:

                df[column] = df[column].fillna(df[column].dropna().mode()[0])
missing_values_test(test)

#Target Feature



train['SalePrice'].describe()
# Correlation heatmap



corr=train_[numerical_feature].corr()

sns.heatmap(corr)
#Separate out Categorical and numerical features

numerical_feature=train_.dtypes[train.dtypes!= 'object'].index

categorical_feature=train_.dtypes[train.dtypes== 'object'].index
numerical_feature.shape[0]
categorical_feature.shape[0]
corr=train_[numerical_feature].corr()

sns.heatmap(corr)
sns.jointplot(x=train_['GrLivArea'], y=train['SalePrice'])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer

#Divide into predictor and target variables

train_X = train.drop('SalePrice', axis=1)

train_y = train.SalePrice

test_X = test
 #one-hot encoding categorical variables 

onehot_train_X = pd.get_dummies(train_X)

onehot_test_X = pd.get_dummies(test_X)

train_X, test_X = onehot_train_X.align(onehot_test_X, join='left', axis=1)

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)

reg = LinearRegression()

cv_scores = cross_val_score(reg, train_X, train_y, cv=5)

print(cv_scores)
reg = LinearRegression()

reg.fit(train_X, train_y)

predictions = reg.predict(test_X)
submission_linreg = pd.DataFrame({'Id': test.Id, 'SalePrice':predictions})
submission_linreg.to_csv('submission_linreg.csv', index=False)

#Drop the id column from the both column

train_p = train

test_p = test

train = train.drop('Id', axis=1)

test = test.drop('Id', axis=1)
#Find Nan values which have more than 40%

threshold=0.4 * len(train)

df=pd.DataFrame(len(train) - train.count(),columns=['count'])

df.index[df['count'] > threshold]



train = train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

test = test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
train['SalePrice'].describe()

#all numeric values

train.select_dtypes(include=np.number).columns


for col in ('MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold'):

    

    train[col] = train[col].fillna(0)

    test[col] = test[col].fillna('0')
#replace NAN values with None

test.select_dtypes(exclude=np.number).columns

for col in ('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition'):

    

    train[col] = train[col].fillna('None')

    test[col] = test[col].fillna('None')
train[train.isnull().any(axis=1)]

test[test.isnull().any(axis=1)]

#Combine the datasets

train_house = train

test_house = test

train_house['train_house']=1 

test_house['test_house']=0
combined=pd.concat([train_house,test_house])

#One hot encoding for for categorical data

ohe_data_frame=pd.get_dummies(combined, 

                           columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition'],

      )
#Splitting the combined dataset after doing OHE .

train_df=ohe_data_frame[ohe_data_frame['train_house']==1]

test_df=ohe_data_frame[ohe_data_frame['train_house']==0]

train_df.drop(['train_house'],axis=1,inplace=True)             

test_df.drop(['train_house','SalePrice'],axis=1,inplace=True) 
train=train_df

test=test_df
X_train = train.drop('SalePrice', axis=1)

Y_train = train['SalePrice']

X_test = test
#GardientBoosting

from sklearn.ensemble import GradientBoostingRegressor



params = {'n_estimators': 3000, 'max_depth': 1, 'min_samples_leaf':15, 'min_samples_split':10, 

          'learning_rate': 0.05, 'loss': 'huber','max_features':'sqrt'}

gbr_model = GradientBoostingRegressor(**params)

gbr_model.fit(X_train, Y_train)
gbr_model.score(X_train, Y_train)

y_grad_predict = gbr_model.predict(X_test)

print(y_grad_predict)
my_submission = pd.DataFrame({'Id': test_p.Id, 'SalePrice': y_grad_predict})

print(my_submission)



my_submission.to_csv('submission2.csv', encoding='utf-8', index=False)
data_to_use = ['Id', 'LotArea', 'OverallQual','OverallCond','YearBuilt',

                  'TotRmsAbvGrd','GarageCars','WoodDeckSF','PoolArea','SalePrice']
data_in_test = data_to_use.copy()

data_in_test.remove("SalePrice")
data_in_test

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", usecols=data_to_use)

df.set_index('Id', inplace=True)

pd.options.display.max_rows=5
df
df.isna().sum().sum()
y = np.log(df.SalePrice)

X = df.drop(['SalePrice'], 1)
from sklearn.model_selection import GridSearchCV

depths = np.arange(1, 21)
num_leafs = [1, 5, 10, 20, 50, 100]
param_grid = [{'decisiontreeregressor__max_depth':depths,

              'decisiontreeregressor__min_samples_leaf':num_leafs}]
from sklearn.pipeline import make_pipeline



pipe_tree = make_pipeline(tree.DecisionTreeRegressor(random_state=1))

from sklearn.metrics import make_scorer

rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

gs = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, scoring=rmse_scorer, cv=10)
