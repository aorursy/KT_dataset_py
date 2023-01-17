from matplotlib import pyplot as plt



import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split



%config InlineBackend.figure_format = 'retina'

%matplotlib inline



plt.style.use('fivethirtyeight')
# read the two data-sets.



test=pd.read_csv('test.csv')

train=pd.read_csv('train.csv')

train.head()

#test.head()
# EDA the data by exploring the shape .

print('the shape of train',train.shape)

print('the shape of test',test.shape)
# explore the data type.

train.dtypes

test.dtypes
# describe the data .

train.describe()

test.describe()
# finding the null values of train .

train.isnull().sum()
# fill the null values .

train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace=True)

train['Alley']=train.Alley.replace(np.nan,'NA')

train['MasVnrType']=train.MasVnrType.replace(np.nan,'CBlock')

train.MasVnrArea.fillna(train.MasVnrArea.mean(), inplace=True)

train['FireplaceQu']=train.FireplaceQu.replace(np.nan,'NA')

train['GarageType']=train.GarageType.replace(np.nan,'NA')

train.GarageYrBlt.fillna(train.GarageYrBlt.mean(), inplace=True)

train['GarageFinish']=train.GarageFinish.replace(np.nan,'NA')

train['GarageQual']=train.GarageQual.replace(np.nan,'NA')

train['GarageCond']=train.GarageCond.replace(np.nan,'NA')

train['PoolQC']=train.PoolQC.replace(np.nan,'NA')

train['Fence']=train.Fence.replace(np.nan,'NA')

train['MiscFeature']=train.MiscFeature.replace(np.nan,'NA')
# check  the rest columns for null values .

print(round((train.isnull().sum()*100/len(train)),1).sort_values(ascending=False).head(7))

# fill the null values .

train['BsmtFinType2']=train.BsmtFinType2.replace(np.nan,'NA')

train['BsmtExposure']=train.BsmtExposure.replace(np.nan,'NA')

train['BsmtQual']=train.BsmtQual.replace(np.nan,'NA')

train['BsmtFinType1']=train.BsmtFinType1.replace(np.nan,'NA')

train['BsmtCond']=train.BsmtCond.replace(np.nan,'NA')

train['Electrical']=train.Electrical.replace(np.nan,'NA')
## finding the null values of test .

print(round((test.isnull().sum()*100/len(test)),1).sort_values(ascending=False).head(35))

# fill the null values .

test['PoolQC']=test.PoolQC.replace(np.nan,'NA')

test['MiscFeature']=test.MiscFeature.replace(np.nan,'NA')

test['Alley']=test.Alley.replace(np.nan,'NA')

test['Fence']=test.Fence.replace(np.nan,'NA')

test['FireplaceQu']=test.FireplaceQu.replace(np.nan,'NA')

test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace=True)

test['GarageQual']=test.GarageQual.replace(np.nan,'NA')

test.GarageYrBlt.fillna(test.GarageYrBlt.mean(), inplace=True)

test['GarageFinish']=test.GarageFinish.replace(np.nan,'NA')

test['GarageCond']=test.GarageCond.replace(np.nan,'NA')

test['GarageType']=test.GarageType.replace(np.nan,'NA')

# fill the null values .

test['BsmtFinType2']=test.BsmtFinType2.replace(np.nan,'NA')

test['BsmtExposure']=test.BsmtExposure.replace(np.nan,'NA')

test['BsmtQual']=test.BsmtQual.replace(np.nan,'NA')

test['BsmtFinType1']=test.BsmtFinType1.replace(np.nan,'NA')

test['BsmtCond']=test.BsmtCond.replace(np.nan,'NA')
print(round((test.isnull().sum()*100/len(test)),1).sort_values(ascending=False).head(18))

test['MasVnrType']=test.MasVnrType.replace(np.nan,'NA')

test['MasVnrType']=test.MasVnrType.replace('None','NA')

test.MasVnrArea.fillna(test.MasVnrArea.mean(), inplace=True)

test['MSZoning']=test.MSZoning.replace(np.nan,'RL')

test['Functional']=test.Functional.replace(np.nan,'Typ')

test['Utilities']=test.Utilities.replace(np.nan,'AllPub')

test['BsmtHalfBath']=test.BsmtHalfBath.replace(np.nan,0.0)

test['BsmtFullBath']=test.BsmtFullBath.replace(np.nan,0.0)

test['KitchenQual']=test.KitchenQual.replace(np.nan,'TA')

test['Exterior2nd']=test.Exterior2nd.replace(np.nan,'VinylSd')

test['Exterior1st']=test.Exterior1st.replace(np.nan,'VinylSd')

test['GarageCars']=test.GarageCars.replace(np.nan,2.0)

test.BsmtFinSF1.fillna(test.BsmtFinSF1.mean(), inplace=True)

test['SaleType']=test.SaleType.replace(np.nan,'WD')

test.TotalBsmtSF.fillna(test.TotalBsmtSF.mean(), inplace=True)

test.BsmtUnfSF.fillna(test.BsmtUnfSF.mean(), inplace=True)

test.BsmtFinSF2.fillna(test.BsmtFinSF2.mean(), inplace=True)

test.GarageArea.fillna(test.GarageArea.mean(), inplace=True)
print(round((test.isnull().sum()*100/len(test)),1).sort_values(ascending=False).head(5))
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
train_d = pd.get_dummies(train)

test_d = pd.get_dummies(test)
x = len(train)

both = pd.concat([train_d.drop('SalePrice',axis=1),test_d],sort=False,ignore_index=True)
both.fillna(0,inplace=True)

both.head()
print(round((both.isnull().sum()*100/len(both)),1).sort_values(ascending=False).head(18))
fig = plt.figure(figsize=(10,5))

ax = fig.gca()



sns.distplot(train_d.SalePrice, bins=30, kde=True , color='orange' )
corr=train.corr(method='kendall')

corr1=corr.nlargest(15,'SalePrice').index

corr1

corr2=train[['SalePrice', 'OverallQual', 'GarageCars', 'GrLivArea', 'FullBath',

       'GarageArea', 'YearBuilt', 'TotalBsmtSF', 'YearRemodAdd', 'Fireplaces',

       '1stFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'OpenPorchSF', 'MasVnrArea']]

corr=corr2.corr()
df1=train[['SalePrice', 'OverallQual', 'GarageCars', 'GrLivArea', 'FullBath',

       'GarageArea', 'YearBuilt', 'TotalBsmtSF', 'YearRemodAdd', 'Fireplaces',

       '1stFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'OpenPorchSF', 'MasVnrArea']]

h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)

sns.despine(left=True, bottom=True)

[x.title.set_size(12) for x in h.ravel()];

[x.yaxis.tick_left() for x in h.ravel()];
f, axes = plt.subplots(1, 2,figsize=(10,5))

sns.boxplot(x=train['1stFlrSF'],y=train['SalePrice'], ax=axes[0])

sns.boxplot(x=train['YearBuilt'],y=train['SalePrice'], ax=axes[1])

sns.despine(left=True, bottom=True)

axes[0].set(xlabel='1stFlrSF', ylabel='SalePrice')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='YearBuilt', ylabel='SalePrice')



f, axe = plt.subplots(1, 1,figsize=(12.18,5))

sns.despine(left=True, bottom=True)

sns.boxplot(x=train['1stFlrSF'],y=train['SalePrice'], ax=axe)

axe.yaxis.tick_left()

axe.set(xlabel='YearBuilt / YearRemodAdd', ylabel='SalePrice');
sns.pairplot(train_d,x_vars=[ 'OverallQual', 'LotFrontage', 'LotArea','GarageArea', '1stFlrSF', 'TotRmsAbvGrd'

       ],y_vars='SalePrice')
mask = np.zeros_like(corr, dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True

fig=plt.figure(figsize=(10,10))

ax = fig.gca()

sns.heatmap(corr, annot=True,ax=ax,mask=mask,linecolor='w',vmin = 0, vmax = +1)

ax.set_title('The correlation between The Features ')

plt.show()
X_train=both[:x]

y_train=train['SalePrice']
X_train.head()
X_test=both[x:]
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
from sklearn.linear_model import LinearRegression
# first model is LinearRegression

model = LinearRegression()



# get cross validated scores

scores = cross_val_score(model, X_train, y_train, cv=5)

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())



# fit and evaluate the data on the whole training set

model.fit(X_train, y_train)

print("Training Score:", model.score(X_train, y_train))
# collect the model coefficients in a dataframe

df_coef = pd.DataFrame(model.coef_, index=X_train.columns,

                       columns=['coefficients'])



# calculate the absolute values of the coefficients

df_coef['coef_abs'] = df_coef.coefficients.abs()

df_coef.head()
# second model is Lasso model 





ls = Lasso(alpha=1316.38)

# evaluate on the training set

scores = cross_val_score(ls, X_train, y_train, cv=5)

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())



# fit and evaluate the data on the whole training set

ls.fit(X_train, y_train)

print("Training Score:", ls.score(X_train, y_train))

y = ls.predict(X_test)

idTest = test['Id']

pd.DataFrame({'Id': idTest, 'SalePrice': y}).to_csv('ls3.csv', index = False)
# third model is Ridge model .





rg = Ridge(alpha=603.1)

# get cross validated scores

scores = cross_val_score(rg, X_train, y_train, cv=5)

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())



# fit and evaluate the data on the whole training set

rg.fit(X_train, y_train)

print("Training Score:", rg.score(X_train, y_train))
# collect the model coefficients in a dataframe

df_coef = pd.DataFrame(rg.coef_, index=X_train.columns,

                       columns=['coefficients'])



# calculate the absolute values of the coefficients

df_coef['coef_abs'] = df_coef.coefficients.abs()

df_coef.head()
y1 = rg.predict(X_test)

idTest = test['Id']

pd.DataFrame({'Id': idTest, 'SalePrice': y1}).to_csv('rig.csv', index = False)
# fourth model is AdaBoostClassifier



from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



classifier = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=7),

    n_estimators=200

)

classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=5)

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())