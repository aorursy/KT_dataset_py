import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

%config InlineBackend.figure_format = 'retina'

%matplotlib inline
train_users = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_users = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print("There were", train_users.shape[0], "observations in the training set and", test_users.shape[0], "in the test set.")

print("In total there were", train_users.shape[0] + test_users.shape[0], "observations.")
plt.figure(figsize=(12,10))

cor = train_users.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train_users.head(10)
test_users.head(10)
train_users.isnull().sum()
SaleCondition_Price=train_users[['SaleCondition','SalePrice']].drop_duplicates()

SaleCondition_Price.groupby(['SaleCondition'])['SalePrice'].aggregate('count').reset_index().sort_values('SalePrice', ascending=False)
YrSold_Price=train_users[['YrSold','SalePrice']].drop_duplicates()

YrSold_Price.groupby(['YrSold'])['SalePrice'].aggregate('count').reset_index().sort_values('YrSold', ascending=False)
def unique_counts(train_users):

   for i in train_users.columns:

       count = train_users[i].nunique()

       print(i, ": ", count)

unique_counts(train_users)
plt.figure(figsize=(12,6))

sns.distplot(train_users.SalePrice.dropna(), rug=True)

sns.despine()
plt.figure(figsize=(12,6))

sns.countplot(x='YrSold', data=train_users)

plt.xlabel('Year')

plt.ylabel('Number of observations')

sns.despine()
train_users.SaleCondition.unique()
plt.figure(figsize=(12,6))

sns.boxplot(y='LotArea', x='SaleCondition',data=train_users)

plt.xlabel('Sale condition')

plt.ylabel('Lot Area')

plt.title('Sale Condition vs. Lot Area')

sns.despine()
train_users.LotFrontage.dropna()

train_users.LotFrontage.describe()
plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="LotFrontage", data=train_users, hue='YrSold', dodge=True)

g.set_title("Lot Frontage Count Distribution by Year", fontsize=20)

g.set_ylabel("Number of Count",fontsize= 17)

g.set_xlabel("Lot Frontage", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_xlim(20,50)

g.set_ylim(0, max(sizes)*1.15)

plt.show()
plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="LandContour", data=train_users, hue='YrSold', dodge=True)

g.set_title("Land Contour in different sold years", fontsize=20)

g.set_ylabel("Number of Count",fontsize= 17)

g.set_xlabel("LandContour", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_ylim(0, max(sizes)*1.15)

plt.show()
plt.figure(figsize=(8,6))



plt.subplot(211)

g = sns.countplot(x="SaleType", data=train_users, hue='YrSold', dodge=True)

g.set_title("Sale Type in different sold years", fontsize=20)

g.set_ylabel("Number of Count",fontsize= 17)

g.set_xlabel("Sale Type", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_ylim(0, max(sizes)*1.15)

plt.show()
plt.figure(figsize=(8,6))



plt.subplot(211)

g = sns.countplot(x="Alley", data=train_users, hue='Street', dodge=True)

g.set_title("Alley and Street", fontsize=20)

g.set_ylabel("Number of Count",fontsize= 17)

g.set_xlabel("Alley", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_ylim(0, max(sizes)*1.15)

plt.show()
data = pd.concat((train_users.loc[:,'MSSubClass':'SaleCondition'],

                      test_users.loc[:,'MSSubClass':'SaleCondition']))
train_users["SalePrice"] = np.log1p(train_users["SalePrice"])

numeric_feats = data.dtypes[data.dtypes != "object"].index

skewed_feats = train_users[numeric_feats].apply(lambda x: skew(x.dropna()))

skewed_feats
skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

data[skewed_feats] = np.log1p(data[skewed_feats])
data = pd.get_dummies(data)

data = data.fillna(data.mean())

X_train = data[:train_users.shape[0]]

X_test = data[train_users.shape[0]:]

y = train_users.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
import xgboost as xgb
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
preds = 0.7*lasso_preds + 0.3*xgb_preds

solution = pd.DataFrame({"id":test_users.Id, "SalePrice":preds})

solution.to_csv("Solution.csv", index = False)