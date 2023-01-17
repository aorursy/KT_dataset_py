import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



from scipy.stats import skew

from scipy.stats.stats import pearsonr



#sklearn

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
train_in = pd.read_csv("../input/train.csv", header = 0, encoding = 'utf-8')
test_in = pd.read_csv("../input/test.csv", header = 0, encoding = 'utf-8')
train = train_in.copy()

test = test_in.copy()
pd.options.display.max_columns = 81
train.head()
train.info()
plt.hist(train['SalePrice'])

plt.xlabel('Sale Price')

plt.ylabel('Frequency')

plt.show
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
corrmat.nlargest(12,'SalePrice')['SalePrice'].index
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

feat_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[feat_cols], size = 2.5)

plt.show();
num_train = train[feat_cols]
num_train.isnull().sum() # no nulls. good. 


null_pc = train.isnull().sum()/len(train) 

null_pc[null_pc > 0.40]
train['FireplaceQu'].value_counts()
train.groupby('FireplaceQu')['SalePrice'].mean().sort_values(ascending=False)
train[train['Fireplaces'] == 0]['FireplaceQu'].isnull().sum()
train[train['FireplaceQu'].isnull()]['Fireplaces'].value_counts()
train['FireplaceQu'].fillna('None', inplace = True)
train.groupby('FireplaceQu')['SalePrice'].mean().sort_values(ascending=False)
null_pc = train.isnull().sum()/len(train) 

null_list = null_pc[null_pc > 0.40].index.tolist()
train.drop(null_list, axis =1 , inplace = True)
#get dummies

obj_train = train.select_dtypes(include = ['object'])
obj_train.head()
obj_train = pd.get_dummies(obj_train)
obj_train.isnull().sum().sum()
y_train = num_train.SalePrice

X_train = np.asmatrix(num_train.drop('SalePrice', axis = 1))
def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))

    rmse = np.mean(rmse)

    return(rmse)
lr = LinearRegression()
lr.fit(X_train, y_train)
rmse_cv(lr)
initial_pred = cross_val_predict(lr, X_train, y_train, cv = 5)
fig, ax = plt.subplots(figsize = (8,4))

ax.scatter(y_train, initial_pred)

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
ridge_mod = Ridge()
rmse_cv(ridge_mod)
ridge_pred = cross_val_predict(ridge_mod, X_train, y_train, cv = 5)
fig, ax = plt.subplots(figsize = (8,4))

ax.scatter(y_train, ridge_pred)

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
skewed_feats = num_train.apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats
skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index
skewed_feats
num_train[skewed_feats] = np.log1p(num_train[skewed_feats])
num_train.head()
fig, ax =plt.subplots(1, 2,figsize = (10,5))

plt.subplot(1, 2, 1)

plt.hist(num_train['SalePrice'])

plt.ylabel('Frequency')

plt.xlabel('Sale Price')

plt.subplot(1, 2, 2)

plt.hist(train['SalePrice']/1e5)

plt.xlabel('Sale Price [1e6]')

plt.ylabel('Frequency')

plt.show
y_train = num_train.SalePrice

X_train = np.asmatrix(num_train.drop('SalePrice', axis = 1))
sk_lr = LinearRegression()
sk_lr.fit(X_train, y_train)
rmse_cv(sk_lr)
skewed_pred = cross_val_predict(lr, X_train, y_train, cv = 5)
fig, ax = plt.subplots(figsize = (8,4))

ax.scatter(y_train, skewed_pred)

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
sk_ridge_mod = Ridge()
sk_ridge_mod.fit(X_train, y_train)
rmse_cv(sk_ridge_mod) 
rmse_cv(sk_ridge_mod)  / rmse_cv(sk_lr)
alphas = [1e-6, 0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")
compl_train = pd.concat([num_train, obj_train], axis = 1)
num_train.shape, compl_train.shape
y_train = compl_train.SalePrice

X_train = compl_train.drop('SalePrice', axis = 1)
clr = LinearRegression()
clr.fit(X_train, y_train)
rmse_cv(clr)
c_ridge = Ridge()
c_ridge.fit(X_train, y_train)
rmse_cv(c_ridge)
c_pred = cross_val_predict(c_ridge, X_train, y_train, cv = 5)
fig, ax = plt.subplots(figsize = (8,4))

ax.scatter(y_train, c_pred)

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
alphas = [1e-6, 0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
lasso_model = LassoCV()
lasso_model.fit(X_train, y_train)
rmse_cv(lasso_model)
model_lasso = LassoCV(alphas = [10, 1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
rmse_cv(model_lasso).mean()
lasso_pred = cross_val_predict(model_lasso, X_train, y_train, cv = 5)
fig, ax = plt.subplots(figsize = (8,4))

ax.scatter(y_train, lasso_pred)

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
lasso_preds = np.expm1(model_lasso.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":lasso_preds})

solution.to_csv("housing_solution1.csv", index = False)