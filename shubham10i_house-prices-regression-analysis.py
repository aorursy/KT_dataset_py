import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.isna().sum()
plt.style.use('fivethirtyeight')
sns.distplot(train['SalePrice'])
corr = train.corr()
corr
k = 10

cols = corr.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.15)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
plt.style.use('dark_background')
cols = ['SalePrice','OverallQual','GrLivArea']
sns.pairplot(train[cols],size=2.5)
train.sort_values(by='GrLivArea',ascending=False)[:2]
# Deleting the top two data
train = train.drop(index = train[train['Id'] == 1299].index)
train = train.drop(index = train[train['Id'] == 524].index)
cols = ['SalePrice','OverallQual','GrLivArea']
sns.pairplot(train[cols],size=2.5)
train_x = train[["OverallQual", "GrLivArea"]]
train_y = train["SalePrice"]
from sklearn.preprocessing import MinMaxScaler
ms_x = MinMaxScaler()
ms_y = MinMaxScaler()
train_y
train_y = train_y.values.reshape(-1,1)
train_y
train_x
train_x = ms_x.fit_transform(train_x)
train_y = ms_y.fit_transform(train_y)
train_x
train_y
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(train_x,train_y)
# partial regression coefficient
print('slope ： {0}'.format(lm.coef_))

# y-intercept
print('y-intercept : {0}'.format(lm.intercept_))

lm_pred = lm.predict(train_x)
lm_pred
from sklearn import metrics
print('MAE :', metrics.mean_absolute_error(train_y, lm_pred))
print('MSE :', metrics.mean_squared_error(train_y, lm_pred))
print('RMSE :', np.sqrt(metrics.mean_squared_error(train_y, lm_pred)))
test_x = test[['OverallQual','GrLivArea']]
test_x
test_x = ms_x.fit_transform(test_x)
test_pred = lm.predict(test_x)
test_pred = ms_y.inverse_transform(test_pred)
test_pred
test["SalePrice"] = test_pred
test[["Id","SalePrice"]].head()
test[["Id","SalePrice"]].to_csv("./submission.csv",index=False)
