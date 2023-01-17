# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import metrics



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR





from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.columns
train_df.info()
train_df.describe()
num_fea = train_df.dtypes[train_df.dtypes!='object'].index

print('Numerical features:', len(num_fea))

cat_fea = train_df.dtypes[train_df.dtypes=='object'].index

print('Categorical features:', len(cat_fea))
all_features = train_df.columns
sns.distplot(train_df['SalePrice'])

plt.title('Distribution of Sales Price')

print("Skewness: %f" % train_df['SalePrice'].skew())

print("Kurtosis: %f" % train_df['SalePrice'].kurt())
train_df['SalePrice_log'] = np.log(train_df['SalePrice'])
sns.distplot(train_df['SalePrice_log'])

plt.title('Distribution of Sales Price with log')

print("Skewness: %f" % train_df['SalePrice_log'].skew())

print("Kurtosis: %f" % train_df['SalePrice_log'].kurt())
def missing_data_train():

    total = train_df.isnull().sum().sort_values(ascending=False)

    percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)

    missing_data_train = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data_train

missing_data_train = missing_data_train()
missing_data_train.head(20)
def missing_data_test():

    total = test_df.isnull().sum().sort_values(ascending=False)

    percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)

    missing_data_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data_test

missing_data_test = missing_data_test()
missing_data_test.head(35)
col_with_fillna = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageType','GarageFinish','GarageQual',

                  'BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtCond','BsmtQual','MasVnrType','Utilities']



for col in col_with_fillna:

      train_df[col] = train_df[col].fillna('None')
def miss():

    total = train_df.isnull().sum().sort_values(ascending=False)

    percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data.head(5)

miss()
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())

train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(train_df['GarageYrBlt'].mode()[0])

train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)

train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])
def miss2():

    total = train_df.isnull().sum().sort_values(ascending=False)

    percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data.head(5)

miss2()
col_with_fillna = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageType','GarageFinish','GarageQual',

                  'BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtCond','BsmtQual','MasVnrType','Utilities']



for col in col_with_fillna:

      test_df[col] = test_df[col].fillna('None')
def miss():

    total = test_df.isnull().sum().sort_values(ascending=False)

    percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data.head(20)

miss()
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())

test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mode()[0])

test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(0)

test_df['MSZoning'] = test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

test_df["Functional"] = test_df["Functional"].fillna("Typ")



test_df['KitchenQual'] = test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])

test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])

test_df['Exterior1st'] = test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])

test_df['SaleType'] = test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])





for col in ('BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'GarageArea', 'GarageCars','TotalBsmtSF', 'BsmtFinSF2', 'BsmtUnfSF'):

    test_df[col] = test_df[col].fillna(0)



def miss():

    total = test_df.isnull().sum().sort_values(ascending=False)

    percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data.head(20)

miss()
def correlation():  

    corr = train_df.corr()

    plt.figure(figsize=(15,10))

    plt.title('Overall Corellation')

    sns.heatmap(corr, annot=False, linewidths=0.5, cmap = 'coolwarm')

correlation()
nr_rows = 12

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



li_num_feats = list(num_fea)

li_not_plot = ['Id', 'SalePrice', 'SalePrice_log']

li_plot_num_feats = [c for c in list(num_fea) if c not in li_not_plot]





for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_plot_num_feats):

            sns.regplot(train_df[li_plot_num_feats[i]], train_df['SalePrice_log'], ax = axs[r][c])                  

plt.tight_layout()    

plt.show()   
num_fea2 = ['SalePrice_log','LotFrontage','LotArea','BsmtFinSF1','OpenPorchSF','OverallQual','YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 

            'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea']            
def heatmap():

    heat_df = train_df[num_fea2]

    corr = heat_df.corr()

    plt.figure(figsize=(15,10))

    plt.title('Overall Corellation')

    sns.heatmap(corr, annot=True, linewidths=0.5, cmap = 'coolwarm') 

heatmap()
li_cat_fea = list(cat_fea)

nr_rows = 15

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_fea):

            sns.boxplot(x=li_cat_fea[i], y=train_df['SalePrice_log'], data=train_df, ax = axs[r][c])

    

plt.tight_layout()    

plt.show()
data = train_df
data = data.drop(['Id', 'SalePrice'], axis=1)
data.head()
data[cat_fea] = data[cat_fea].apply(LabelEncoder().fit_transform)
data.head()
X = data.loc[:,data.columns!= 'SalePrice_log']

X.head()
y = data.loc[:,data.columns== 'SalePrice_log']

y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(X_train.shape)

print(y_train.shape)
print(X_test.shape)

print(y_test.shape)
dtr = DecisionTreeRegressor(random_state=1)

dtr.fit(X_train,y_train)
y_pred_test = dtr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_test))

print('MSE:', metrics.mean_squared_error(y_test, y_pred_test))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
rf = RandomForestRegressor(random_state=2)

rf.fit(X_train,y_train)
y_pred_test1 = rf.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_test1))

print('MSE:', metrics.mean_squared_error(y_test, y_pred_test1))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test1)))
linreg1 = LinearRegression()

linreg1.fit(X_train,y_train)
y_pred_test2 = linreg1.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_test2))

print('MSE:', metrics.mean_squared_error(y_test, y_pred_test2))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test2)))
dtree_class = DecisionTreeRegressor(random_state=4)

parameters = {'max_depth' : [7,8,9,10] , 'max_features' : [11,12,13,14] ,

               'max_leaf_nodes' : [None, 12,15,18,20] ,'min_samples_split' : [20,25,30],

                'presort': [False,True] , 'random_state': [5]}

dtree_class = GridSearchCV(dtree_class, parameters)

dtree_class.fit(X_train, y_train)



y_pred_test3 = dtree_class.predict(X_test)
print("Mean cross-validated score of the best_estimator : ", dtree_class.best_score_)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_test3))

print('MSE:', metrics.mean_squared_error(y_test, y_pred_test3))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test3)))
plt.scatter(y_pred_test2, y_test, c = "blue",  label = "Training data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()