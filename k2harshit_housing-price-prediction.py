import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
style.use('ggplot')
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info(), train.shape, test.shape
# NANs
total = train.isnull().sum().sort_values(ascending=False)
percent = train.isnull().sum().sort_values(ascending=False)
train.isnull().count().sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Missing', 'Percent'])
missing.head(20)
train = train.drop((missing[missing['Missing'] > 1]).index, 1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
# check if any NANs remaining
train.isnull().sum().max()
mis = missing[missing['Missing'] > 1]
mis
# droping the columns in test set that were droped in training set
for i in test.columns:    
    if i in mis.index:
        test.drop(i, 1, inplace=True)
# remaining missing values in test set
test.isnull().sum().sort_values(ascending=False)[:15]
# remove missing values from test set
miss = test.isnull().sum().sort_values(ascending=False)[:15].index
for i in miss:
    if test[i].dtype != 'object':
        test[i].fillna(test[i].median(), inplace=True)
    else:  # if test[i].dtype == 'object'
        test[i].fillna(test[i].mode()[0], inplace=True)
fig, ax = plt.subplots()
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
ax1.set_title('With Outliers')
sns.regplot(data=train, x='GrLivArea',y='SalePrice', ax=ax1)
ax2.set_title('Without Outliers')
sns.regplot(data=train[train.GrLivArea < 4500], x='GrLivArea',y='SalePrice', ax=ax2)
bonf_outlier = [88,462,523,588,632,968,1298,1324]
train = train.drop(bonf_outlier)
sns.distplot(train['SalePrice'])
print('Skewness: ', train['SalePrice'].skew())
# take log transform
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'])
# removing skewness from numerical variables
from scipy.stats import skew
from scipy.special import boxcox1p

numeric_feats = train.dtypes[train.dtypes != "object"].index
numeric_feats
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
# train[skewed_feats] = np.log1p(train[skewed_feats])
fitted_lambda = 0.15
for j in skewed_feats:
    train[j] = boxcox1p(train[j], fitted_lambda)
# doing the same for test set
numeric_feats = test.dtypes[train.dtypes != "object"].index
numeric_feats
skewed_feats = test[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
# test[skewed_feats] = np.log1p(test[skewed_feats])
for j in skewed_feats:
    test[j] = boxcox1p(test[j], fitted_lambda)
f, ax = plt.subplots(figsize=(12, 9)) 
sns.heatmap(train.corr(), cmap="YlGnBu")
c = train.corr()
c['SalePrice'].sort_values(ascending=False)[:10]
# convert categorical variable into dummy
train = pd.get_dummies(train)
test = pd.get_dummies(test)
# Ensure the test data is encoded in the same manner as the training data
final_train, final_test = train.align(test, join='inner', axis=1)  # inner join

X_train = final_train.drop('Id', axis=1)
y_train = train['SalePrice']
X_test = final_test.drop('Id', axis=1)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg.score(X_train, y_train)
linrid = Ridge(alpha=20.0)
linrid.fit(X_train, y_train)
linrid.score(X_train, y_train)
linlasso = Lasso(alpha=0.5, max_iter=10000)
linlasso.fit(X_train, y_train)
linlasso.score(X_train, y_train)
rid = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
rid.fit(X_train, y_train)
y_pred = rid.predict(X_test)
alpha = rid.alpha_
print('best alpha',alpha)
rid.score(X_train, y_train)
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
model_xgb.score(X_train, y_train)
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': y_pred*10000  # temporary fix
})
submission.to_csv('house.csv', index=False)
