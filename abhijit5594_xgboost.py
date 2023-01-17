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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
import statistics
import scipy.stats as stats
from scipy.stats import norm, skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, plot_importance 
## Import Trainning data. 
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_df.head()
test_df.head()
data_info = open("../input/house-prices-advanced-regression-techniques/data_description.txt", "r")
for i in data_info:
    print(i)
train_df.describe().T
test_df.describe().T
train_df.columns
#train_df.drop(columns = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','Id'],axis = 1, inplace = True)
missing_count = train_df.isnull().sum().sort_values(ascending = False)
missing_perc = (missing_count/len(train_df['Id']))*100
missing_count
missing_perc
train_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','Id'],axis = 1, inplace = True)
test_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','Id'],axis = 1, inplace = True)
sns.distplot(train_df['SalePrice'], fit = norm)

(mu, sigma) = norm.fit(train_df['SalePrice'])

plt.legend(['Normal Distribution ($\mu=${:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc = 'best')
plt.ylabel('Frequency')
plt.title('Sale Price distribution')

fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot = plt)
plt.show()
sns.distplot(np.log1p(train_df['SalePrice']), fit = norm)

(mu, sigma) = norm.fit(np.log1p(train_df['SalePrice']))

plt.legend(['Normal Distribution ($\mu=${:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc = 'best')
plt.ylabel('Frequency')
plt.title('Sale Price distribution')

fig = plt.figure()
res = stats.probplot(np.log1p(train_df['SalePrice']), plot = plt)
plt.show()
# reading data
pd.set_option('precision', 2)
plt.figure(figsize = (10, 8))
sns.heatmap(train_df.drop(['SalePrice'], axis  = 1).corr(), square = True)
plt.suptitle("Pearson Correlation Heatmap")
plt.show();
corr_with_sale_price = train_df.corr()["SalePrice"].sort_values(ascending=False)
plt.figure(figsize=(14,6))
corr_with_sale_price.drop("SalePrice").plot.bar()
plt.show();
sns.pairplot(train_df[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']])
plt.show();
train_df.drop(['1stFlrSF', 'GarageCars'],axis = 1, inplace = True)
test_df.drop(['1stFlrSF', 'GarageCars'],axis = 1, inplace = True)

train_df.shape
test_df.shape
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

#log transform skewed numeric features:
numeric_feats = train_df.dtypes[train_df.dtypes != "object"].index
numeric_feats
skewed_feats = train_df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats
skewed_feats = skewed_feats.index
skewed_feats
train_df[skewed_feats] = np.log1p(train_df[skewed_feats])
train_df.head(2)
test_df[skewed_feats] = np.log1p(test_df[skewed_feats])
test_df.head(2)
Y = train_df['SalePrice']
train_df.drop(['SalePrice'], axis = 1, inplace = True)
final_data=pd.concat([train_df,test_df])
final_data.shape
final_data = pd.get_dummies(final_data)
final_data = final_data.fillna(final_data.mean())

train_data=final_data.iloc[:1460,:]
test_data=final_data.iloc[1460:,:]
X_train, X_test, y_train, y_test  = train_test_split(train_data, Y, test_size = 0.2, random_state = 0)
xgb_model4 = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                         colsample_bytree=1, max_depth=7, n_jobs=-1)
xgb_model4.fit(X_train,y_train)
y_train_pred4 = xgb_model4.predict(X_train)
y_pred4 = xgb_model4.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred4, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred4))
train_mse4 = mean_squared_error(y_train_pred4, y_train)
test_mse4 = mean_squared_error(y_pred4, y_test)
train_rmse4 = np.sqrt(train_mse4)
test_rmse4 = np.sqrt(test_mse4)
print('Train RMSE: %.4f' % train_rmse4)
print('Test RMSE: %.4f' % test_rmse4)
train_sales = np.expm1(y_train_pred4)
test_sale = np.expm1(y_pred4)
val_pred = xgb_model4.predict(test_data)
val_sale = np.expm1(val_pred)

submission_file = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
submission_file.head(2)

#make the submission data frame
submission = {
    'Id': submission_file['Id'].values,
    'SalePrice': val_sale
}
solution = pd.DataFrame(submission)
solution.head()
solution.to_csv('submission.csv',index=False)