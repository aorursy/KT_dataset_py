# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

import seaborn as sns

from statsmodels.api import OLS

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.decomposition import PCA

import folium

import eli5

import datetime as dt

%matplotlib inline
train = pd.read_csv('/kaggle/input/diamonds-price/diamonds_train.csv')

test_df = pd.read_csv('/kaggle/input/diamonds-price/diamonds_test.csv')
train.head()
test_df.head()
train_desc = train.describe()

train_desc
train.info()
train.isna().sum()
#price iqr

desc = train.describe()

iqr = desc.iloc[6,1] - desc.iloc[4,1]

upper_bound = desc.iloc[6,1] + (1.5*iqr)
train_cut = train.groupby('cut')

train_cut_desc = train_cut['price'].describe()

train_cut_desc
outliers = train[train['price'] > upper_bound]

outliers
out_cut = outliers.groupby('cut')

out_cut_desc = out_cut['price'].describe()

out_cut_desc
plt.figure(figsize=(16,10))

plt.subplot(221)

plt.title('Price Mean (Outliers)')

plt.plot(out_cut_desc.index.tolist(), out_cut_desc['mean'])

plt.subplot(222)

plt.title('Sample (Outliers)')

plt.plot(out_cut_desc.index.tolist(), out_cut_desc['count'], 'orange')

plt.subplot(223)

plt.title('Price Mean (All)')

plt.plot(out_cut_desc.index.tolist(), train_cut_desc['mean'])

plt.subplot(224)

plt.title('Sample (All)')

plt.plot(out_cut_desc.index.tolist(), train_cut_desc['count'], 'orange')
train_clean = train.drop(outliers.index.tolist(), axis=0)

train_clean
train_clean_cut = train_clean.groupby('cut')
train
train_desc
train_clean_desc = train_clean.describe()

train_clean_desc
#x = length

#y = width

#z = depth

train[(train['x'] == 0) | (train['y'] == 0)]
train_clean[(train_clean['x'] == 0) | (train_clean['y'] == 0)]
train.drop(train[train['x'] == 0].index, axis=0, inplace=True)

train_clean.drop(train_clean[train_clean['x'] == 0].index, axis=0, inplace=True)
print('Length of train data: ', len(train))

print('Length of train_clean data: ', len(train_clean))
train.cut.unique()
#group again after dropping impossible value

train_cut = train.groupby('cut')

train_clean_cut = train_clean.groupby('cut')
train_cut.first()
#distribution with outliers

plt.figure(figsize=(16,10))

plt.subplot(231)

plt.title('All Price Distribution')

sns.distplot(train.price)

plt.subplot(232)

plt.title('Fair Cut Price Distribution')

sns.distplot(train_cut.get_group('Fair')['price'])

plt.subplot(233)

plt.title('Good Cut Price Distribution')

sns.distplot(train_cut.get_group('Good')['price'])

plt.subplot(234)

plt.title('Very Good Cut Price Distribution')

sns.distplot(train_cut.get_group('Very Good')['price'])

plt.subplot(235)

plt.title('Premium Cut Price Distribution')

sns.distplot(train_cut.get_group('Premium')['price'])

plt.subplot(236)

plt.title('Ideal Cut Price Distribution')

sns.distplot(train_cut.get_group('Ideal')['price'])
#distribution without outliers

plt.figure(figsize=(16,10))

plt.subplot(231)

plt.title('All Price Distribution')

sns.distplot(train_clean['price'])

plt.subplot(232)

plt.title('Fair Cut Price Distribution')

sns.distplot(train_clean_cut.get_group('Fair')['price'])

plt.subplot(233)

plt.title('Good Cut Price Distribution')

sns.distplot(train_clean_cut.get_group('Good')['price'])

plt.subplot(234)

plt.title('Very Good Cut Price Distribution')

sns.distplot(train_clean_cut.get_group('Very Good')['price'])

plt.subplot(235)

plt.title('Premium Cut Price Distribution')

sns.distplot(train_clean_cut.get_group('Premium')['price'])

plt.subplot(236)

plt.title('Ideal Cut Price Distribution')

sns.distplot(train_clean_cut.get_group('Ideal')['price'])
plt.figure(figsize=(10,6))

sns.boxplot(y='price', x='cut', hue='cut', data=train)
plt.figure(figsize=(10,6))

sns.boxplot(y='price', x='cut', hue='cut', data=train_clean)
#price correlation with carat and cutting quality (w/ outliers)

plt.figure(figsize=(10,6))

sns.scatterplot(x='carat',y="price",hue="cut",palette="Set2",data=train)
#price correlation with carat and cutting quality (w/o outliers)

plt.figure(figsize=(10,6))

sns.scatterplot(x='carat',y="price",hue="cut",palette="Set2",data=train_clean)
#pearson correlation w/ outliers

train_cor = train.iloc[:,1:].corr()

plt.figure(figsize=(10,8))

sns.heatmap(train_cor, annot=True)

plt.show()
#pearson correlation w/ outliers

train_clean_cor = train_clean.iloc[:,1:].corr()

plt.figure(figsize=(10,8))

sns.heatmap(train_cor, annot=True)

plt.show()
price_corr = train_cor['price']

price_clean_corr = train_clean_cor['price']



plt.figure(figsize=(10,8))

plt.plot(price_corr.index, price_corr)

plt.plot(price_corr.index, price_clean_corr)

plt.legend(['w/ outliers', 'w/o outliers'])

plt.show()
train_cut['cut'].describe()
colors=['#ff9999','#66b3ff','#E66032', '#99ff99','#ffcc99']



plt.figure(figsize=(16,8))

plt.subplot(121)

plt.title('Cutting Percentage (w/ outliers)')

plt.pie(train_cut['cut'].describe()['count'], labels=train_cut['cut'].describe().index,

        colors=colors, autopct='%1.1f%%', startangle=90)



plt.subplot(122)

plt.title('Cutting Percentage (w/o outliers)')

plt.pie(train_clean_cut['cut'].describe()['count'], labels=train_clean_cut['cut'].describe().index,

        colors=colors, autopct='%1.1f%%', startangle=90)



plt.show()
train.head()
train_clean.head()
train.drop(['depth', 'table'], axis=1, inplace=True)

train_clean.drop(['depth', 'table'], axis=1, inplace=True)
train1 = train.copy()

train1.head()
train2 = train_clean.copy()

train2.head()
train1 = pd.get_dummies(train1, prefix=['cut', 'color', 'clarity'])

train2 = pd.get_dummies(train2, prefix=['cut', 'color', 'clarity'])
id_train = train['id']

id_clean = train_clean['id']

train.drop(['id'],axis=1,inplace=True)

train_clean.drop(['id'],axis=1,inplace=True)

train1.drop(['id'],axis=1,inplace=True)

train2.drop(['id'],axis=1,inplace=True)
train1.head()
train2.head()
#w/ outliers

X = train1.drop('price', axis=1)

y = train1['price']

ols = sm.OLS(y, sm.add_constant(X))

results = ols.fit()

results.summary()
sns.distplot(results.resid)
# alpha= 0.05

# Null-hypo: Autocorrelation is absent

# Alternative Hypothesis: Autocorrelation is present



from statsmodels.stats import diagnostic



diagnostic.acorr_ljungbox(results.resid, lags =1, return_df=True)
# Using Goldfeld Quandt we test for Heteroscedasticity

# alpha= 0.05

# Null Hypo: Error terms are homoscedastic

# Alt. Hypo: Error terms are heteroscedastic



import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(results.resid, results.model.exog)

lzip(name, test)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])]

pd.Series(vif, index = X.columns, name = 'VIF')
#w/ outliers

X = train2.drop('price', axis=1)

y = train2['price']

ols = sm.OLS(y, sm.add_constant(X))

results2 = ols.fit()

results2.summary()
sns.distplot(results2.resid)
# alpha= 0.05

# Null-hypo: Autocorrelation is absent

# Alternative Hypothesis: Autocorrelation is present



from statsmodels.stats import diagnostic



diagnostic.acorr_ljungbox(results2.resid, lags =1, return_df=True)
# Using Goldfeld Quandt we test for Heteroscedasticity

# alpha= 0.05

# Null Hypo: Error terms are homoscedastic

# Alt. Hypo: Error terms are heteroscedastic



import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(results2.resid, results2.model.exog)

lzip(name, test)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])]

pd.Series(vif, index = X.columns, name = 'VIF')
def evaluate(model, test_features, test_labels):

    vals = dict()

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    rmse = np.sqrt(mean_squared_error(test_labels, predictions))

    r2 = r2_score(test_labels, predictions)

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    print('RMSE = {:0.2f}'.format(rmse))

    print('R2 Score = {:0.2f}.'.format(r2))

    vals['accuracy'] = accuracy

    vals['rmse'] = rmse

    vals['r2'] = r2

    

    return vals
#w/ outliers

X = train1.drop('price', axis=1)

y = train1['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



#w/o outliers

X2 = train2.drop('price', axis=1)

y2 = train2['price']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)
lr = LinearRegression()

lr.fit(X_train, y_train)

lr_eval1 = evaluate(lr, X_test, y_test)
lr2 = LinearRegression()

lr2.fit(X2_train, y2_train)

lr_eval2 = evaluate(lr2,X2_test, y2_test)
# print(len(train2))

# print(len(train_dum))

# train_dum.tail(1)

train2 = train2.set_index(np.arange(0,len(train2),1))

train2.tail()
scaler = MinMaxScaler()

scaler.fit(train2)

scaled = scaler.transform(train2)

dfscaled = pd.DataFrame(scaled, columns=train2.columns)

dfscaled['price'] = train2['price']

dfscaled.head()
X = dfscaled.drop('price', axis=1)

y = dfscaled['price']

pca = PCA(n_components=10)

pca.fit(X)

xpca = pca.transform(X)

xpca
np.sum(pca.explained_variance_ratio_)
dfpca = pd.DataFrame(xpca, columns=['pc'+str(i) for i in range(1,11)])

dfpca['price'] = dfscaled['price']

dfpca.head()
print('train2 length: ', len(train2))

print('dfscaled length: ', len(dfscaled))

print('dfpca length: ', len(dfpca))
print('train2 length: ', train2.isna().any().any())

print('dfscaled length: ', dfscaled.isna().any().any())

print('dfpca length: ', dfpca.isna().any().any())
plt.figure(figsize=(16,8))

sns.heatmap(dfpca.corr(), annot=True)

plt.show()
Xp = dfpca.drop('price',axis=1)

yp = dfpca['price']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp)
lr3 = LinearRegression()

lr3.fit(Xp_train, yp_train)

lr_eval3 = evaluate(lr3, Xp_test, yp_test)
scaledcon = dfscaled[['carat', 'x', 'y', 'z', 'price']]

scaledcon.head()
scaledcon.describe()
Xcon = scaledcon.drop('price',axis=1)

ycon = scaledcon['price']

pcac = PCA(n_components=2)

pcac.fit(Xcon)

con_pca= pcac.transform(Xcon)

con_pca
pcac.explained_variance_ratio_
dfpcac = pd.DataFrame(con_pca, columns=['pc1', 'pc2'])

dfpcac['price'] = scaledcon['price']

dfpcac.head()
# plt.figure(figsize=(10,8))

sns.heatmap(dfpcac.corr(), annot=True)

plt.show()
dfpcac['price']
Xpc = dfpcac.drop('price', axis=1)

ypc = dfpcac['price']

Xpc_train, Xpc_test, ypc_train, ypc_test = train_test_split(Xpc, ypc, test_size=0.2)
lr4 = LinearRegression()

lr4.fit(Xpc_train, ypc_train)

lr_eval4 = evaluate(lr4, Xpc_test, ypc_test)
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 20)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 20]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4, 6, 8, 10]

# Method of selecting samples for training each tree

bootstrap = [True, False]



random_grid = {

    'n_estimators': n_estimators,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'bootstrap': bootstrap

}



print(random_grid)
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,

                               cv=5, verbose=2, random_state=41, n_jobs=-1)



rf_random.fit(X2_train, y2_train)
rf_random.best_params_
base_rf = RandomForestRegressor()

base_rf.fit(X2_train, y2_train)

base_rf_eval1 = evaluate(base_rf, X2_test, y2_test)
best_rf = rf_random.best_estimator_

best_rf.fit(X2_train, y2_train)

best_rf_eval2 = evaluate(best_rf, X2_test, y2_test)
best_rf_eval3 = evaluate(best_rf, X2_train, y2_train)
boost_grid = {

    "loss": ['ls', 'lad', 'huber', 'quantile'],

    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],

    "n_estimators": np.arange(100,201,20),

    "subsample": np.linspace(0.5,1,6),

    "min_samples_split": [2,4,6,8,10,12],

    "min_samples_leaf": [1,2,3,4,5,6,7,8],

    "max_depth": [ 3, 4, 5, 6, 8, 10, 12, 15],

    "max_features": ['auto', 'sqrt', 'log2'],

    "alpha": np.linspace(0.1,1,10),

    "max_leaf_nodes": [10,20,30,40,50,60,70,'None'],

    "warm_start": [True, False],

    "validation_fraction": np.linspace(0,1,10)

}



boost_grid
gbr = GradientBoostingRegressor()

gbr_random = RandomizedSearchCV(estimator=gbr, param_distributions=boost_grid, n_iter=100,

                               cv=5, verbose=2, random_state=41, n_jobs=-1)



gbr_random.fit(X2_train, y2_train)
gbr_random.best_params_
base_gbr = GradientBoostingRegressor()

base_gbr.fit(X2_train, y2_train)

basegbr_eval = evaluate(base_gbr, X2_test, y2_test)
best_gbr = gbr_random.best_estimator_

best_gbr.fit(X2_train, y2_train)

bestgbr_eval = evaluate(best_gbr, X2_test, y2_test)
bestgbr_eval_train = evaluate(best_gbr, X2_train, y2_train)
bestrf_scores = cross_val_score(best_rf, X2, y2, cv=5, scoring='neg_root_mean_squared_error')

bestrf_scores
plt.plot(np.array(range(1,6)), np.abs(bestrf_scores), 'bo-')

plt.show()
bestgbr_scores = cross_val_score(best_gbr, X2, y2, cv=5, scoring='neg_root_mean_squared_error')

bestgbr_scores
plt.plot(np.array(range(1,6)), np.abs(bestgbr_scores), 'bo-')

plt.show()
rmse_vals = [lr_eval2['rmse'], lr_eval3['rmse'], lr_eval4['rmse'], best_rf_eval2['rmse'], bestgbr_eval['rmse']]

r2_vals = [lr_eval2['r2'], lr_eval3['r2'], lr_eval4['r2'], best_rf_eval2['r2'], bestgbr_eval['r2']]

df_eval = pd.DataFrame({'rmse': rmse_vals, 'r2': r2_vals}, index=['LinearRegression', 'LinearRegression w/ pca', 'LinearRegression w/ pca (only continous)', 'RandomForest', 'GradientBoost'])

df_eval
plt.figure(figsize=(10,5))

plt.plot(df_eval.index, df_eval['rmse'])

plt.xticks(rotation=45)

plt.grid()

plt.show()
X2_test.columns
test_df.head()
test_dum = pd.get_dummies(test_df)

test_dum
test_id = test_dum['id']

test_dum.drop('id', axis=1, inplace=True)
test_dum.drop(['depth', 'table'], axis=1, inplace=True)
print(X2_test.columns)

print(test_dum.columns)
test_dum['price'] = best_gbr.predict(test_dum)
test_dum['id'] = test_id
output = test_dum[['id', 'price']]
output.to_csv('submission.csv', index=False)

output.head()