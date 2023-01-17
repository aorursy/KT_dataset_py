# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.describe()
train
test
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])

sns.distplot(np.log(train['SalePrice']))
plt.boxplot(train['SalePrice'])



# there are some outliers here i guess
numeric_features = train.select_dtypes(include=[np.int, np.float])
numeric_features.columns
cat_features = train.drop(numeric_features.columns, axis=1)
cat_features.shape
numeric_features.shape
#Dropping the column ID

numeric_features.drop('Id', inplace=True, axis= 1)
numeric_features
numeric_features.isnull().any()
['GarageYrBlt','MasVnrArea','LotFrontage']
for i in numeric_features.columns:

    print(numeric_features[i].nunique())
discrete_var = []

for i in numeric_features.columns:

    if numeric_features[i].nunique()<= 15:

        print("{}  has the \n{}".format(i,numeric_features[i].value_counts()))
# we will drop PoolArea as most of the values are zero

numeric_features.shape



numeric_features.corr()['SalePrice'].sort_values(ascending=True)
f, ax =plt.subplots(figsize = (16,12))

sns.heatmap(numeric_features.corr(), vmax = 1, square = True)
numeric_features[['GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','SalePrice']].corr()
# Lets see correlation one more time , Here will emphasis on the variables which are more corelated to target columns and we will do 

#
highcorr_numfr = list(numeric_features.corr()['SalePrice'].sort_values(ascending = False)[0:10].index.drop(['GarageArea','1stFlrSF']))
highcorr_numfr


#sns.pairplot(numeric_features[highcorr_numfr])
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

sale_norm = ss.fit_transform(numeric_features['SalePrice'].values.reshape(-1,1))
sale_norm[sale_norm.reshape(1,-1).argsort()][0][0:5]
sale_norm[sale_norm.reshape(1,-1).argsort()][0][-5:]
# For GrLivArea, TotalBsmtSF, we have to chek with SalePrice
#f, ax = plt.figure(figsize=(16,12))

sns.scatterplot(numeric_features['GrLivArea'], numeric_features['SalePrice'])
numeric_features.drop(numeric_features['GrLivArea'].argsort()[-2:], inplace=True)
numeric_features
# Lets plot again and see ouliers are gone or not 

sns.scatterplot(numeric_features['GrLivArea'], numeric_features['SalePrice'])
# Lets look the outlier of Total BsmtSF

sns.scatterplot(numeric_features['TotalBsmtSF'], numeric_features['SalePrice'])
figure = plt.figure()

fig = stats.probplot(numeric_features['TotalBsmtSF'], plot = plt)

plt.show()
# I will delete the two data row which are outliers in 'GrLivAres', from data which will be used for categorical data analysis

train.drop(numeric_features['GrLivArea'].argsort()[-2:], inplace=True)
cat_features.columns
# Lets get the categorical data from train data set

cat_features = train[cat_features.columns]
cat_features.info()


# Lets see null how many percent is the null values of the data

null_percent =(cat_features.isnull().sum()/cat_features.isnull().count()).sort_values(ascending= False)*100



null_percent.index[:14]
cat_features.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',

       'GarageQual', 'GarageFinish', 'GarageType', 'BsmtFinType2',

       'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond'], axis=1,inplace=True)
cat_features.isnull().sum().sort_values()
# There are now two cloumns which contains the null values and less than 9 row, So wildrop those row

cat_features.dropna(axis=0, inplace=True)
# Lets calculate the number unique of categories predent inside each column

n_unique = {}

for i in cat_features.columns:

    n_unique[i] = cat_features[i].nunique()
n_unique
cat_features['MSZoning'].value_counts()[:3]
d = {}

for i in cat_features.columns:

    d[i] =cat_features[i].value_counts()[:3]

    
d
for i in ['ExterQual','ExterCond','HeatingQC','KitchenQual']:

    print(cat_features[i].value_counts())
cat_features[['ExterQual','ExterCond','HeatingQC','KitchenQual']]

def encoder(x):

    if x == 'Ex':

        return 4

    elif x == 'Gd':

        return 3

    elif x == 'TA':

        return 2

    elif x == 'Fa':

        return 1

    elif x == 'PO' or x == 'Po':

        return 0

    else:

        return x
# We have done lable encoder manually on the columns

for i in ['ExterQual','ExterCond','HeatingQC','KitchenQual']:

    cat_features[i] = cat_features[i].map(encoder)
# We have converted this 'Y' and 'N' into 1 and 0

cat_features['CentralAir'] =cat_features['CentralAir'].map(lambda x: 1 if x=='Y'else 0)
# Now deal with Paved Drive which has three categories

def ypn(x):

    if x=='Y':

        return 2

    elif x == 'P':

        return 1

    elif x == 'N':

        return 0

    else:

        return x
cat_features['PavedDrive'] = cat_features['PavedDrive'].map(ypn)
cat_features[['ExterQual','ExterCond','HeatingQC','KitchenQual','PavedDrive','CentralAir']]
# First get the coulmn name which are remained to preprocess

remaining = cat_features.columns.drop(['ExterQual','ExterCond','HeatingQC','KitchenQual','PavedDrive','CentralAir'])
remaining
dummies = pd.get_dummies(cat_features[remaining])
dummies
# We will join the cat_features and dummies

cat_features = cat_features.join(other = dummies)
# Lets drop the columns , from which we have got the dummies i.e. remaining

cat_features.drop(remaining, inplace= True, axis=1)
numeric_features
cat_features
final_train =cat_features.join(other = numeric_features, on = cat_features.index, how= 'inner')
final_train.fillna
# Still there are some null values in LotFrontage,GarageYrBlt, Which we havenot checked in numerica feature exploration, mistake happend, Lets correct now

final_train['LotFrontage']=final_train['LotFrontage'].fillna(final_train['LotFrontage'].median())
final_train['GarageYrBlt'] = final_train['GarageYrBlt'].fillna(final_train['GarageYrBlt'].median())
final_train[['GarageYrBlt','MasVnrArea','LotFrontage']].isna().sum()
final_train
test.drop(['Id','PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',

       'GarageQual', 'GarageFinish', 'GarageType', 'BsmtFinType2',

       'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond'], axis=1,inplace=True)
test.isnull().sum()
train['MSZoning'].value_counts()
test['MSZoning'] = test['MSZoning'].fillna('RL')

# Filling the 'RL' value in missing place , RL is most common
train['LotArea'].median()
train1 = train.copy()
train
#remove outliers

train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)

train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)

train.reset_index(drop=True, inplace=True)
train1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#remove outliers

train1.drop(train1[(train1['OverallQual']<5) & (train1['SalePrice']>200000)].index, inplace=True)

train1.drop(train1[(train1['GrLivArea']>4500) & (train1['SalePrice']<300000)].index, inplace=True)

train1.reset_index(drop=True, inplace=True)
train_labels = train1['SalePrice']
full_features = pd.concat((train1,test1))

full_features.drop('SalePrice', axis=1, inplace=True)
full_features.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',

       'GarageQual', 'GarageFinish', 'GarageType', 'BsmtFinType2',

       'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond'], axis=1,inplace=True)
for i in ['ExterQual','ExterCond','HeatingQC','KitchenQual']:

    full_features[i] = full_features[i].map(encoder)
full_features['CentralAir'] = full_features['CentralAir'].map(lambda x: 1 if x=='Y'else 0)
full_features['PavedDrive'] = full_features['PavedDrive'].map(ypn)
full_features.isnull().sum()
full_features[numeric_features.columns.drop('SalePrice')].isnull().sum()
numeric_features.columns.drop('SalePrice')
full_features[numeric_features.columns.drop('SalePrice')].isnull().sum()
for i in numeric_features.columns.drop('SalePrice'):

    full_features[i] = full_features[i].fillna(train[i].median())
full_features['SaleType'].value_counts()
full_features.isna().sum()
full_features['MSZoning'].isna().sum()
full_features['MSZoning'].value_counts().index[0]
full_features['MSZoning'].fillna(full_features['MSZoning'].value_counts().index[0]).isna().sum()
for i in full_features.columns:

    full_features[i]=full_features[i].fillna(full_features[i].value_counts().index[0])
full_features = pd.get_dummies(full_features).reset_index(drop = True)
full_features.drop('Id',axis = 1, inplace=True)
train2 = full_features.iloc[:len(train1),:]
train2['SalePrice'] = train1['SalePrice']
test2 = full_features.iloc[len(train1):,:]
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.model_selection import KFold, GridSearchCV, train_test_split,cross_val_score

from sklearn.metrics import mean_squared_error

X = train2.drop('SalePrice', axis=1)

y = train2['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

r_scale = RobustScaler().fit(X_train)

X_train_scale = r_scale.transform(X_train)

X_test_scale = r_scale.transform(X_test)

lr = LinearRegression().fit(X_train_scale, y_train)

y_pred = lr.predict(X_test_scale)

mean_squared_error(y_test, y_pred)
# I will use cross_validation, of the 10 folds 

kfold = KFold(n_splits = 10, random_state=42, shuffle=True)
tree_reg = RandomForestRegressor()
#parameters for the Grid Search CV

para = {'n_estimators':[700,500,200],'max_depth':[50,80]}

forest_reg = GridSearchCV(tree_reg, param_grid=para, cv= kfold)
forest_reg.fit(X, y)
forest_reg.best_params_
forest_reg.best_score_
y_pred2 = forest_reg.predict(test2)
output = pd.DataFrame({'Id':test1['Id'],'SalePrice':y_pred2})
output.to_csv('random_forest_solution', index= False)