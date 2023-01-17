# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col=0)

test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col=0)





print("train: ", train.shape)

print("test: ", test.shape)

train.head()
X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0)

y = train[['SalePrice']]
X.info()
numeric_ = X.select_dtypes(exclude=['object']).drop(['MSSubClass'],axis = 1).copy()

numeric_
disc_num_var = ['OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',

                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']

cont_num_var = []

for i in numeric_.columns:

    if i not in disc_num_var:

        cont_num_var.append(i)
categor_train = X.select_dtypes(['object']).copy()

categor_train
fig = plt.figure(figsize=(18,16))

for index,col in enumerate(cont_num_var):

    plt.subplot(6,4,index+1)

    sns.distplot(numeric_.loc[:,col].dropna(), kde = False)

fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(14,15))

for index,col in enumerate(cont_num_var):

    plt.subplot(6,4,index+1)

    sns.boxplot(y=col, data=numeric_.dropna())

fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(25,25))

for index, cols in enumerate(disc_num_var):

    plt.subplot(5,3,index+1)

    sns.countplot(x=col,data = numeric_.dropna())

fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(20,20))

for index in range(len(categor_train.columns)):

    plt.subplot(9,5,index+1)

    sns.countplot(x=categor_train.iloc[:,index], data=categor_train.dropna())

    plt.xticks(rotation=90)

fig.tight_layout(pad=1.0)
plt.figure(figsize=(14,12))



sns.heatmap(numeric_.corr(),annot=True, mask = numeric_.corr() < 0.8 ,linewidth=0.7,cmap='Blues')
numeric_train = train.select_dtypes(exclude=['object'])

correlation = numeric_train.corr()

correlation[['SalePrice']].sort_values(['SalePrice'],ascending = False)
X.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)
plt.figure(figsize=(25,8))



plt.title('Number of missing rows')





missing_count = pd.DataFrame(X.isnull().sum(), columns = ['sum']).sort_values(by=['sum'],ascending = False).head(15).reset_index()



missing_count.columns = ['features','sum']



sns.barplot(x = 'features',y = 'sum',data = missing_count)
X.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)
print(correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False).tail(10))



X.drop(['MoSold','YrSold'], axis=1, inplace=True)
cat_col = X.select_dtypes(include=['object']).columns

overfit_cat1 = []

for i in cat_col:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 96:

        overfit_cat1.append(i)

overfit_cat1 = list(overfit_cat1)

X.drop(overfit_cat1, axis=1,inplace = True)
num_col = X.select_dtypes(exclude = ['object']).columns

overfit_num = []

for i in num_col:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros/len(X)*100 > 96:

        overfit_num.append(i)

overfit_num = list(overfit_num)

X.drop(overfit_num,axis=1,inplace = True)
X.shape
print("Categorical Features with >96% of the same value: ",overfit_cat1)

print("Numerical Features with >96% of the same value: ",overfit_num)
cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1', 

       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',"KitchenQual",

       "HeatingQC",'ExterQual','ExterCond']



X[cat] = X[cat].fillna("NA")

cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]

X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))

X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))

X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



#numerical

cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]

X[cont] = X[cont] = X[cont].fillna(X[cont].mean())
X['MSSubClass'] = X['MSSubClass'].apply(str)

X['BsmtFinType1'].value_counts()
ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}

fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}

expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}

fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}
ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']

for col in ord_col:

    X[col] = X[col].map(ordinal_map)

    

fin_col = ['BsmtFinType1','BsmtFinType2']

for col in fin_col:

    X[col] = X[col].map(fintype_map)



X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)

X['Fence'] = X['Fence'].map(fence_map)

train.shape
test.shape
X.shape

X.drop('BsmtFinSF2', axis=1,inplace = True)
X.shape
X = pd.get_dummies(X)

X.shape
plt.figure(figsize=(10,6))

plt.title("Before transformation of SalePrice")

dist = sns.distplot(train['SalePrice'],norm_hist=False)
y["SalePrice"] = np.log(y['SalePrice'])
x = X.loc[train.index]

y = y.loc[train.index]
X.shape
test = X.loc[test.index]
test.shape
from sklearn.preprocessing import RobustScaler



cols = x.select_dtypes(np.number).columns

transformer = RobustScaler().fit(x[cols])

x[cols] = transformer.transform(x[cols])

test[cols] = transformer.transform(test[cols])
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2020)
y.shape
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

from sklearn import ensemble

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_val_score

from catboost import CatBoostRegressor
xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror',n_estimators = 1000,

                  learning_rate =0.05) 

xgb.fit(X_train,y_train)
lgbm = LGBMRegressor(boosting_type='gbdt',objective='regression', max_depth=8,

                    lambda_l1=0.0001, lambda_l2=0, learning_rate=0.1,

                    n_estimators=1000, max_bin=200, min_child_samples=20, 

                    bagging_fraction=0.75, bagging_freq=5,

                    bagging_seed=7, feature_fraction=0.8,

                    feature_fraction_seed=7, verbose=-1)

lgbm.fit(X_train,y_train)
cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent',

                       n_estimators = 1000,learning_rate=0.05)

cb.fit(X_train,y_train)
def blend_models_predict(X, b, c, d):

    return ((b* xgb.predict(X)) + (c * lgbm.predict(X)) + (d * cb.predict(X)))
subm = np.exp(blend_models_predict(test, 0.4, 0.2, 0.4))

submission = pd.DataFrame({'Id': test.index,

                           'SalePrice': subm})



submission.to_csv("../../kaggle/working/submission.csv", index=False)