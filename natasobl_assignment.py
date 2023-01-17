# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
train_file = "../input/train.csv"
test_file = "../input/test.csv"
train = pd.read_csv(train_file)
train.head()
f=train['GrLivArea']
f
test = pd.read_csv(test_file)
test.head()
train.SalePrice.describe()
plt.hist(np.log(train.SalePrice))
train.LotConfig.value_counts()
train.Foundation.value_counts()
import seaborn as sns
sns.countplot(train.LotConfig)
sns.countplot(train.Foundation)
train['tr_street'] = pd.get_dummies(train.Street, drop_first=True)
test['tr_street'] = pd.get_dummies(train.Street, drop_first=True)
train.tr_street.value_counts()
def encode(x): return 1 if x == 'PConc' or x== 'CBlock' else 0
train['enc_foundation'] = train.Foundation.apply(encode)
test['enc_foundation'] = test.Foundation.apply(encode)
train.enc_foundation.value_counts()
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()

corr['SalePrice'].sort_values(ascending=False)
plt.plot(train.OverallQual, train.SalePrice, '.')
 
quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)
quality_pivot

quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
target = np.log(train.SalePrice)
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel(' gr living area')
plt.show()

plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel(' garage area')
plt.show()
train = train[train['GrLivArea'] < 4000]

plt.scatter(x=train['GrLivArea'], y=np.log(train.SalePrice))
plt.xlim(-200,4500) 
plt.ylabel('Sale Price')
plt.xlabel('GrLivArea')
plt.show()

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 ", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE ', mean_squared_error(y_test, predictions))
submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)

submission['SalePrice'] = final_predictions
submission.head()
submission.to_csv('submission1.csv', index=False)

