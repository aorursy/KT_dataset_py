import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

%matplotlib inline

sns.set_style("whitegrid")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
print("Train shape: ", train.shape)

print("Test shape: ", test.shape)
X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0)

y = train[['SalePrice']]
X.describe()
X.head()
X.tail()
numerical_features = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()

numerical_features.columns
cat_num_var = ['OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',

                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

num_var = [feature for feature in numerical_features if feature not in cat_num_var]   

num_var             
cat_features = X.select_dtypes(include=[np.object]).copy()

cat_features['MSSubClass'] = X['MSSubClass'] 



cat_features.columns
y.describe()
sns.distplot(y)
print("Skewness ", y.skew())

print("Kurtosis ", y.kurt())
for feature in num_var:

    data = pd.concat([y,X[feature]], axis = 1)

    data.plot.scatter(x = feature, y = 'SalePrice', ylim = (0, 800000))
fig = plt.figure(figsize=(20,15))

for feature in cat_num_var:

    data = pd.concat([y,X[feature]], axis = 1)

    data.plot.scatter(x = feature, y = 'SalePrice', ylim = (0, 800000))
for feature in cat_features:

    data = pd.concat([y,X[feature]], axis = 1)

    f, ax = plt.subplots(figsize = (8, 6))

    fig = sns.boxplot(x = feature, y = 'SalePrice', data = data)

    fig.axis(ymin=0, ymax=800000)
fig = plt.figure(figsize=(10,6))

for index,col in enumerate(num_var):

    plt.subplot(6,4,index+1)

    sns.distplot(numerical_features.loc[:,col].dropna(), kde=False)

fig.tight_layout(pad=1.0)

fig = plt.figure(figsize=(20,15))

for index,col in enumerate(cat_num_var):

    plt.subplot(9,5,index+1)

    sns.countplot(x=col, data=numerical_features.dropna())

fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(20,15))

for index in range(len(cat_features.columns)):

    plt.subplot(9,5,index+1)

    sns.countplot(x=cat_features.iloc[:,index], data=cat_features.dropna())

fig.tight_layout(pad=1.0)
corr_matrix = X.corr()

f, ax = plt.subplots(figsize=(20,15))

sns.heatmap(corr_matrix,  linewidth=0.5, cmap='Greys')
corr_matrix = train.corr()

f, ax = plt.subplots(figsize=(10,8))

num_of_var = 15 

cols = corr_matrix.nlargest(num_of_var, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1)

heatmap = sns.heatmap(cm,vmin = -1, vmax = 1, center = 0, annot = True,cmap = 'Accent', yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','GarageArea','1stFlrSF', 'TotRmsAbvGrd','YearRemodAdd']

sns.pairplot(train[cols], size = 2.5)

plt.show()
total = X.isnull().sum().sort_values(ascending=False)

percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
X.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)
X.drop(['GarageYrBlt','GrLivArea','TotalBsmtSF','GarageCars'], axis=1, inplace=True)
corr_matrix[['SalePrice']].sort_values(['SalePrice'], ascending=False).tail(10)



X.drop(['MoSold','YrSold'], axis=1, inplace=True)
cat_col = X.select_dtypes(include=['object']).columns

overfit_cat = []

for i in cat_col:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 90:

        overfit_cat.append(i)



overfit_cat = list(overfit_cat)

X = X.drop(overfit_cat, axis=1)
num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns

overfit_num = []

for i in num_col:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 90:

        overfit_num.append(i)



overfit_num = list(overfit_num)

X = X.drop(overfit_num, axis=1)
pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(20)
cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1', 

       'GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',"KitchenQual",

       "HeatingQC",'ExterQual','ExterCond']



X[cat] = X[cat].fillna("NA")
cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType"]

X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
print("Mean of LotFrontage: ", X['LotFrontage'].mean())

print("Mean of GarageArea: ", X['GarageArea'].mean())

neigh_lot = X.groupby('Neighborhood')['LotFrontage'].mean().reset_index(name='LotFrontage_mean')

neigh_garage = X.groupby('Neighborhood')['GarageArea'].mean().reset_index(name='GarageArea_mean')



fig, axes = plt.subplots(1,2,figsize=(22,8))

axes[0].tick_params(axis='x', rotation=90)

sns.barplot(x='Neighborhood', y='LotFrontage_mean', data=neigh_lot, ax=axes[0])

axes[1].tick_params(axis='x', rotation=90)

sns.barplot(x='Neighborhood', y='GarageArea_mean', data=neigh_garage, ax=axes[1])



X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))

X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))





cont = [ "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",  "MasVnrArea"]

X[cont] = X[cont] = X[cont].fillna(X[cont].mean())
X['MSSubClass'] = X['MSSubClass'].apply(str)
X.isnull().sum().max()
ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}

fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}

expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}

fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}
ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual', 'FireplaceQu']

for col in ord_col:

    X[col] = X[col].map(ordinal_map)

    

fin_col = ['BsmtFinType1','BsmtFinType2']

for col in fin_col:

    X[col] = X[col].map(fintype_map)



X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)

X['Fence'] = X['Fence'].map(fence_map)
out_col = ['LotFrontage','LotArea','BsmtFinSF1']

fig = plt.figure(figsize=(20,5))

for index,col in enumerate(out_col):

    plt.subplot(1,5,index+1)

    sns.boxplot(y=col, data=X)

fig.tight_layout(pad=1.5)
train = train.drop(train[train['LotFrontage'] > 200].index)

train = train.drop(train[train['LotArea'] > 100000].index)

train = train.drop(train[train['BsmtFinSF1'] > 4000].index)

train = train.drop(train[train['TotalBsmtSF'] > 5000].index)

train = train.drop(train[train['GrLivArea'] > 4000].index)

X['TotalLot'] = X['LotFrontage'] + X['LotArea']

X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']

X['TotalBath'] = X['FullBath'] + X['HalfBath']

X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch']
columns = ['MasVnrArea','TotalBsmtFin','2ndFlrSF','WoodDeckSF','TotalPorch']



for col in columns:

    col_name = col+'_bin'

    X[col_name] = X[col].apply(lambda x: 1 if x > 0 else 0)
X = pd.get_dummies(X)
from sklearn.preprocessing import RobustScaler



cols = X.select_dtypes(np.number).columns

X[cols] = RobustScaler().fit_transform(X[cols])
sns.distplot(train['SalePrice'], fit=norm)

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
y["SalePrice"] = np.log(y['SalePrice'])
x = X.loc[train.index]

y = y.loc[train.index]

test = X.loc[test.index]
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_val, y_val)])
print ("R^2 is: \n", my_model.score(X_val, y_val))
predictions = my_model.predict(X_val)
from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_val, predictions))
final_predictions = my_model.predict(test)

final_predictions = np.exp(final_predictions)
print ("Original predictions are: \n", predictions[:5], "\n")

print ("Final predictions are: \n", final_predictions[:5])
submission = pd.DataFrame({'Id': test.index,

                           'SalePrice': final_predictions})



submission.to_csv("submission1.csv", index=False)