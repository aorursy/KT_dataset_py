# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
import matplotlib.pyplot as plt
import seaborn as sns
print(test_data.shape, train_data.shape)
train_data.dtypes
train_data.SalePrice.describe()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(train_data.SalePrice, bins=50)

plt.subplot(1,2,2)
sns.distplot(np.log1p(train_data.SalePrice), bins=50)
plt.title('Log')

plt.tight_layout()
print("SKew: ", train_data.SalePrice.skew())
print("Kurt: ", train_data.SalePrice.kurt())
train_data["GrLivArea"]
gr = 'GrLivArea'
concat = pd.concat([train_data["SalePrice"], train_data[gr]], axis=1)
concat.head()
sns.set()
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[columns], size = 2.5)
plt.show()
target = train_data['SalePrice']
t_l= np.log1p(train_data['SalePrice'])
train_data = train_data.drop(["SalePrice"], axis=1)
train_test = pd.concat([train_data, test_data], ignore_index=True)
columns_cat = [column for column in train_test.columns.values if train_test[column].dtype == 'object']
categories = train_test[columns_cat]
numeric = train_test.drop(columns_cat, axis=1)
numeric.head()
categories.head()
from scipy.stats import skew
num_skew = numeric.apply(lambda x: skew(x.dropna()))
num_skew = num_skew[num_skew > .75]
numeric[num_skew.index] = np.log1p(numeric[num_skew.index])
num_skew
length_num = numeric.shape[0]
for col in numeric.columns.values:
    missing_values = numeric[col].isnull().sum()
    if(missing_values > 50):
        numeric = numeric.drop(col, axis = 1)
    else:
        numeric = numeric.fillna(numeric[col].median())
length_cat = categories.shape[0]
for col in categories.columns.values:
    missing_values = categories[col].isnull().sum()
    if missing_values > 50:
        print("{}".format(col))
        categories.drop(col, axis = 1)
    else:
        pass
categories.describe()
numeric.describe()
cat_dummies = pd.get_dummies(categories)
cat_dummies
print("Numeric shape: ", numeric.shape)
print("Categories shape: ", categories.shape)
data = pd.concat([numeric, cat_dummies], axis=1)
data.head(3)
train_data
test_data
train_data = data.iloc[:1460 - 1]
train_data = train_data.join(t_l)
test_data = data.iloc[1459 + 1:]
train_data
test_data
X_train = train_data[train_data.columns.values[1:-1]]
y_train = train_data[train_data.columns.values[-1]]
X_test = test_data[test_data.columns.values[1:]]
def rmse_(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 7))
    return(rmse)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train)
random_forest = RandomForestRegressor(n_estimators=300, n_jobs=-1)
random_forest.fit(X_train1, y_train1)
y_pred = random_forest.predict(X_test1)
random_forest.fit(X_train, y_train)
y_p_log = random_forest.predict(X_test)
submission = pd.DataFrame({'Id':test_data['Id'], 'SalePrice':np.expm1(y_p_log)})
submission
submission.to_csv("rf.csv", index=False)
from sklearn.model_selection import cross_val_score
rmse_(random_forest).mean()
from sklearn.ensemble import GradientBoostingRegressor
grad_boost = GradientBoostingRegressor()
grad_boost.fit(X_train, y_train)
rmse_(grad_boost).mean()
y_p_log = grad_boost.predict(X_test)
submission = pd.DataFrame({'Id':test_data['Id'], 'SalePrice':np.expm1(y_p_log)})
submission
submission.to_csv("grad_boost.csv", index=False)
from sklearn.svm import SVR
svm_reg=SVR(kernel='linear',degree=1)
svm_reg.fit(X_train,y_train)
y_p_log=svm_reg.predict(X_test)
rmse_(svm_reg)
submission = pd.DataFrame({'Id':test_data['Id'], 'SalePrice':np.expm1(y_p_log)})
submission
submission.to_csv("svr.csv", index=False)