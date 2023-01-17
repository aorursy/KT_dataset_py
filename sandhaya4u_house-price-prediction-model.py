import os

print(os.listdir("../input"))
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from matplotlib import rcParams

import xgboost as xgb

%matplotlib inline 

sns.set_style('whitegrid')
import scipy.stats as stats

from scipy import stats

from scipy.stats import pointbiserialr, spearmanr, skew, pearsonr
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.linear_model import Ridge, RidgeCV, LassoCV

from sklearn import linear_model
house_train = pd.read_csv("../input/train.csv")

house_test = pd.read_csv("../input/test.csv")
house_train.shape, house_test.shape
house_train.head()
house_test.head()
# "Descriptive Statistics": Summary of Target Variable

house_train['SalePrice'].describe()
# Let's plot histogram to check data is normally distributed or not?

fig, ax = plt.subplots(figsize=(12, 8))

sns.distplot(house_train['SalePrice'])
#skewness and kurtosis

print("Skewness: %f" % house_train['SalePrice'].skew())

print("Kurtosis: %f" % house_train['SalePrice'].kurt())
house_train.info()
house_train.describe(include='all')
#correlation matrix

c_mat = house_train.corr()

f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(c_mat, square=True)
# Highly Correlated Features or Variables

c_mat = house_train.corr()

top_corr_features = c_mat.index[abs(c_mat["SalePrice"])>0.4]

plt.figure(figsize=(10,10))

g = sns.heatmap(house_train[top_corr_features].corr(),annot=True)
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(house_train[cols], size = 2.5)

plt.show()
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#Box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90)
house_train[['OverallQual','SalePrice']].groupby(['OverallQual'],

as_index=False).mean().sort_values(by='OverallQual', ascending=False)
house_train[['GarageCars','SalePrice']].groupby(['GarageCars'],

as_index=False).mean().sort_values(by='GarageCars', ascending=False)
house_train[['Fireplaces','SalePrice']].groupby(['Fireplaces'],

as_index=False).mean().sort_values(by='Fireplaces', ascending=False)
house_train.isnull().sum().sort_values(ascending=False).head(20)
#plot of missing value features

plt.figure(figsize=(12, 8))

sns.heatmap(house_train.isnull())

plt.show()
house_test.isnull().sum().sort_values(ascending=False).head(20)
#plot of missing value features

plt.figure(figsize=(12, 8))

sns.heatmap(house_test.isnull())

plt.show()
total = house_train.isnull().sum().sort_values(ascending=False)

percent = (house_train.isnull().sum()/house_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
total = house_test.isnull().sum().sort_values(ascending=False)

percent = (house_test.isnull().sum()/house_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#Create a list of column to fill NA with "None" or 0.

to_null = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',

           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'BsmtFullBath', 'BsmtHalfBath',

           'PoolQC', 'Fence', 'MiscFeature']

for col in to_null:

    if house_train[col].dtype == 'object':



        house_train[col].fillna('None',inplace=True)

        house_test[col].fillna('None',inplace=True)

    else:



        house_train[col].fillna(0,inplace=True)

        house_test[col].fillna(0,inplace=True)
#Fill NA with common values.

house_test.loc[house_test.KitchenQual.isnull(), 'KitchenQual'] = 'TA'

house_test.loc[house_test.MSZoning.isnull(), 'MSZoning'] = 'RL'

house_test.loc[house_test.Utilities.isnull(), 'Utilities'] = 'AllPub'

house_test.loc[house_test.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'

house_test.loc[house_test.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'

house_test.loc[house_test.Functional.isnull(), 'Functional'] = 'Typ'

house_test.loc[house_test.SaleType.isnull(), 'SaleType'] = 'WD'

house_train.loc[house_train['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

house_train.loc[house_train['LotFrontage'].isnull(), 'LotFrontage'] = house_train['LotFrontage'].mean()

house_test.loc[house_test['LotFrontage'].isnull(), 'LotFrontage'] = house_test['LotFrontage'].mean()
house_train.loc[house_train.MasVnrType == 'None', 'MasVnrArea'] = 0

house_test.loc[house_test.MasVnrType == 'None', 'MasVnrArea'] = 0

house_test.loc[house_test.BsmtFinType1=='None', 'BsmtFinSF1'] = 0

house_test.loc[house_test.BsmtFinType2=='None', 'BsmtFinSF2'] = 0

house_test.loc[house_test.BsmtQual=='None', 'BsmtUnfSF'] = 0

house_test.loc[house_test.BsmtQual=='None', 'TotalBsmtSF'] = 0
#Let's check again is there any missing values present in data or not

house_train.columns[house_train.isnull().any()]

plt.figure(figsize=(10, 5))

sns.heatmap(house_train.isnull())
house_test.loc[house_test.GarageCars.isnull(), 'GarageCars'] = 0

house_test.loc[house_test.GarageArea.isnull(), 'GarageArea'] = 0
#Let's check again is there any missing values present in data or not

house_test.columns[house_test.isnull().any()]

plt.figure(figsize=(10, 5))

sns.heatmap(house_test.isnull())
total = house_test.isnull().sum().sort_values(ascending=False)

percent = (house_test.isnull().sum()/house_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(30)
corr = house_train.corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1)
threshold = 0.8 # Threshold value.

def correlation():

    for i in house_train.columns:

        for j in house_train.columns[list(house_train.columns).index(i) + 1:]: #Ugly, but works. This way there won't be repetitions.

            if house_train[i].dtype != 'object' and house_train[j].dtype != 'object':

                #pearson is used by default for numerical.

                if abs(pearsonr(house_train[i], house_train[j])[0]) >= threshold:

                    yield (pearsonr(house_train[i], house_train[j])[0], i, j)

            else:

                #spearman works for categorical.

                if abs(spearmanr(house_train[i], house_train[j])[0]) >= threshold:

                    yield (spearmanr(house_train[i], house_train[j])[0], i, j)
corr_list = list(correlation())

corr_list
#It seems that SalePrice is skewered, so it needs to be transformed.

sns.distplot(house_train['SalePrice'], kde=False, color='c', hist_kws={'alpha': 0.9})
#As expected price rises with the quality.

sns.regplot(x='OverallQual', y='SalePrice', data=house_train, color='Orange')
#Price also varies depending on neighborhood.

plt.figure(figsize = (12, 6))

sns.boxplot(x='Neighborhood', y='SalePrice',  data=house_train)

xt = plt.xticks(rotation=30)
#There are many little houses.

plt.figure(figsize = (12, 6))

sns.countplot(x='HouseStyle', data=house_train)

xt = plt.xticks(rotation=30)
#And most of the houses are single-family, so it isn't surprising that most of the them aren't large.

sns.countplot(x='BldgType', data=house_train)

xt = plt.xticks(rotation=30)
#Most of fireplaces are of good or average quality. And nearly half of houses don't have fireplaces at all.

pd.crosstab(house_train.Fireplaces, house_train.FireplaceQu)
sns.factorplot('HeatingQC', 'SalePrice', hue='CentralAir', data=house_train)

sns.factorplot('Heating', 'SalePrice', hue='CentralAir', data=house_train)
#One more interesting point is that while pavement road access is valued more, for alley they quality isn't that important.

fig, ax = plt.subplots(1, 2, figsize = (12, 5))

sns.boxplot(x='Street', y='SalePrice', data=house_train, ax=ax[0])

sns.boxplot(x='Alley', y='SalePrice', data=house_train, ax=ax[1])
#We can say that while quality is normally distributed, overall condition of houses is mainly average.

fig, ax = plt.subplots(1, 2, figsize = (12, 5))

sns.countplot(x='OverallCond', data=house_train, ax=ax[0])

sns.countplot(x='OverallQual', data=house_train, ax=ax[1])
fig, ax = plt.subplots(2, 3, figsize = (16, 12))

ax[0,0].set_title('Gable')

ax[0,1].set_title('Hip')

ax[0,2].set_title('Gambrel')

ax[1,0].set_title('Mansard')

ax[1,1].set_title('Flat')

ax[1,2].set_title('Shed')

sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Gable'], jitter=True, ax=ax[0,0])

sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Hip'], jitter=True, ax=ax[0,1])

sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Gambrel'], jitter=True, ax=ax[0,2])

sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Mansard'], jitter=True, ax=ax[1,0])

sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Flat'], jitter=True, ax=ax[1,1])

sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Shed'], jitter=True, ax=ax[1,2])
sns.stripplot(x="GarageQual", y="SalePrice", data=house_train, hue='GarageFinish', jitter=True)
sns.pointplot(x="PoolArea", y="SalePrice", hue="PoolQC", data=house_train)
#There is only one such pool and sale condition for it is 'Abnorml'.

house_train.loc[house_train.PoolArea == 555]
fig, ax = plt.subplots(1, 2, figsize = (12, 5))

sns.stripplot(x="SaleType", y="SalePrice", data=house_train, jitter=True, ax=ax[0])

sns.stripplot(x="SaleCondition", y="SalePrice", data=house_train, jitter=True, ax=ax[1])
#MSSubClass shows codes for the type of dwelling, it is clearly a categorical variable.

house_train['MSSubClass'].unique()
house_train['MSSubClass'] = house_train['MSSubClass'].astype(str)

house_test['MSSubClass'] = house_test['MSSubClass'].astype(str)
for col in house_train.columns:

    if house_train[col].dtype != 'object':

        if skew(house_train[col]) > 0.75:

            house_train[col] = np.log1p(house_train[col])

        pass

    else:

        dummies = pd.get_dummies(house_train[col], drop_first=False)

        dummies = dummies.add_prefix("{}_".format(col))

        house_train.drop(col, axis=1, inplace=True)

        house_train = house_train.join(dummies)

        

for col in house_test.columns:

    if house_test[col].dtype != 'object':

        if skew(house_test[col]) > 0.75:

            house_test[col] = np.log1p(house_test[col])

        pass

    else:

        dummies = pd.get_dummies(house_test[col], drop_first=False)

        dummies = dummies.add_prefix("{}_".format(col))

        house_test.drop(col, axis=1, inplace=True)

        house_test = house_test.join(dummies)
#This is how the data looks like now.

house_train.head()
# Spilit training and testing dataset

X_train = house_train.drop('SalePrice',axis=1)

Y_train = house_train['SalePrice']

X_test  = house_test
#Function to measure accuracy.

def rmlse(val, target):

    return np.sqrt(np.sum(((np.log1p(val) - np.log1p(np.expm1(target)))**2) / len(target)))
Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, Y_train, test_size=0.30)
ridge = Ridge(alpha=10, solver='auto').fit(Xtrain, ytrain)

val_ridge = np.expm1(ridge.predict(Xtest))

rmlse(val_ridge, ytest)
ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))

ridge_cv.fit(Xtrain, ytrain)

val_ridge_cv = np.expm1(ridge_cv.predict(Xtest))

rmlse(val_ridge_cv, ytest)
las = linear_model.Lasso(alpha=0.0005).fit(Xtrain, ytrain)

las_ridge = np.expm1(las.predict(Xtest))

rmlse(las_ridge, ytest)
las_cv = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))

las_cv.fit(Xtrain, ytrain)

val_las_cv = np.expm1(las_cv.predict(Xtest))

rmlse(val_las_cv, ytest)
model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2) #the params were tuned using xgb.cv

model_xgb.fit(Xtrain, ytrain)

xgb_preds = np.expm1(model_xgb.predict(Xtest))

rmlse(xgb_preds, ytest)
forest = RandomForestRegressor(min_samples_split =5,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = None,

                                max_depth = None,

                                n_estimators = 300,

                                max_features = 'auto')



forest.fit(Xtrain, ytrain)

Y_pred_RF = np.expm1(forest.predict(Xtest))

rmlse(Y_pred_RF, ytest)
coef = pd.Series(las_cv.coef_, index = X_train.columns)

v = coef.loc[las_cv.coef_ != 0].count() 

print('So we have ' + str(v) + ' variables')
#Basically I sort features by weights and take variables with max weights.

indices = np.argsort(abs(las_cv.coef_))[::-1][0:v]
#Features to be used. I do this because I want to see how good will other models perform with these features.

features = X_train.columns[indices]

for i in features:

    if i not in X_test.columns:

        print(i)
X_test['RoofMatl_ClyTile'] = 0
X = X_train[features]

Xt = X_test[features]
Xtrain1, Xtest1, ytrain1, ytest1 = train_test_split(X, Y_train, test_size=0.33)
ridge = Ridge(alpha=5, solver='svd').fit(Xtrain1, ytrain1)

val_ridge = np.expm1(ridge.predict(Xtest1))

rmlse(val_ridge, ytest1)
las_cv = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10)).fit(Xtrain1, ytrain1)

val_las = np.expm1(las_cv.predict(Xtest1))

rmlse(val_las, ytest1)
model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2) #the params were tuned using xgb.cv

model_xgb.fit(Xtrain1, ytrain1)

xgb_preds = np.expm1(model_xgb.predict(Xtest1))

rmlse(xgb_preds, ytest1)
forest = RandomForestRegressor(min_samples_split =5,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = None,

                                max_depth = 100,

                                n_estimators = 300,

                                max_features = None)



forest.fit(Xtrain1, ytrain1)

Y_pred_RF = np.expm1(forest.predict(Xtest1))

rmlse(Y_pred_RF, ytest1)
las_cv1 = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))

las_cv1.fit(X, Y_train)

lasso_preds = np.expm1(las_cv1.predict(Xt))
#I added XGBoost as it usually improves the predictions.

model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.1)

model_xgb.fit(X, Y_train)

xgb_preds = np.expm1(model_xgb.predict(Xt))
preds = 0.7 * lasso_preds + 0.3 * xgb_preds
submission = pd.DataFrame({

        'Id': house_test['Id'].astype(int),

        'SalePrice': preds

    })

submission.to_csv('home.csv', index=False)
model_lasso = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 100))

model_lasso.fit(X_train, Y_train)

coef = pd.Series(model_lasso.coef_, index = X_train.columns)

v1 = coef.loc[model_lasso.coef_ != 0].count()

print('So we have ' + str(v1) + ' variables')
indices = np.argsort(abs(model_lasso.coef_))[::-1][0:v1]

features_f=X_train.columns[indices]
print('Features in full, but not in val:')

for i in features_f:

    if i not in features:

        print(i)

print('\n' + 'Features in val, but not in full:')

for i in features:

    if i not in features_f:

        print(i)
for i in features_f:

    if i not in X_test.columns:

        X_test[i] = 0

        print(i)

X = X_train[features_f]

Xt = X_test[features_f]
model_lasso = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))

model_lasso.fit(X, Y_train)

lasso_preds = np.expm1(model_lasso.predict(Xt))
model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X, Y_train)

xgb_preds = np.expm1(model_xgb.predict(Xt))
solution = pd.DataFrame({"id":house_test.Id, "SalePrice":0.7*lasso_preds + 0.3*xgb_preds})

solution.to_csv("House_price.csv", index = False)