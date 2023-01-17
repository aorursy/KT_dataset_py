import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Graph display setting
%matplotlib inline
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
train.SalePrice.describe()
sns.distplot(train.SalePrice)
# correlation coefficient
corr = train.corr()
corr.head()
# The number of columns to be displayed in the heat map
k = 10

# Calculate for the top 10 columns with the highest correlation with SalesPrice
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)

# Font size of the heatmap
sns.set(font_scale=1.25)

# View in a heat map
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# scatter plots
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea']
sns.pairplot(train[cols], size = 2.5)
plt.show()
# top two highest-ranked data
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
# Deleting the top two data
train = train.drop(index = train[train['Id'] == 1299].index)
train = train.drop(index = train[train['Id'] == 524].index)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea']
sns.pairplot(train[cols], size = 2.5)
plt.show()
# feature values
train_x = train[["OverallQual", "GrLivArea"]]
train_y = train["SalePrice"]
from sklearn.preprocessing import StandardScaler
# We are going to scale to data
train_y = train_y.values.reshape(-1,1)
# Data scaling
scaler_x = StandardScaler()
scaler_y = StandardScaler()

train_x = scaler_x.fit_transform(train_x)
train_y = scaler_y.fit_transform(train_y)
train_x
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(train_x, train_y)

# partial regression coefficient
print('slope ： {0}'.format(lm.coef_))

# y-intercept
print('y-intercept : {0}'.format(lm.intercept_))
lm_pred = lm.predict(train_x)
lm_pred
plt.figure(figsize = (15,8))
plt.scatter(train_y, lm_pred)
plt.xlabel('train_y')
plt.ylabel('lm_pred_Y')
plt.show()
plt.figure(figsize = (16,8))
plt.plot(train_y, label = 'train_y')
plt.plot(lm_pred, label = 'lm_pred_Y')
plt.show()
from sklearn import metrics
print('MAE :', metrics.mean_absolute_error(train_y, lm_pred))
print('MSE :', metrics.mean_squared_error(train_y, lm_pred))
print('RMSE :', np.sqrt(metrics.mean_squared_error(train_y, lm_pred)))
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()
test_x = test[["OverallQual", "GrLivArea"]]
test_x = scaler_x.fit_transform(test_x)
test_pred = lm.predict(test_x)
test_pred = scaler_y.inverse_transform(test_pred)
test_pred
test["SalePrice"] = test_pred
test[["Id","SalePrice"]]
test[["Id","SalePrice"]].to_csv("./submission.csv",index=False)