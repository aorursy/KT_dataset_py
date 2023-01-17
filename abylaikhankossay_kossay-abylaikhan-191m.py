# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

dataset
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train
X_train = train.iloc[:, :-1].values

y_train = train.iloc[:, -1].values
X_train
y_train
from sklearn.model_selection import train_test_split
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(20,15))

ax.scatter(train['SalePrice'], train['LotArea'])

ax.set_xlabel('Sale Price')

ax.set_ylabel('Lot Area')

plt.show()
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
train.SalePrice.describe()
print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])
train.OverallQual.unique()
quality_pivot = train.pivot_table(index='OverallQual',

                  values='SalePrice', aggfunc=np.median)

quality_pivot
quality_pivot.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))

plt.xlim(-200,1600) # This forces the same scale as before

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
print ("Unique :", train.MiscFeature.unique())
categoricals = train.select_dtypes(exclude=[np.number])

categoricals.describe()
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=42, test_size=.33)
from sklearn import linear_model

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

print ('RMSE: \n', mean_squared_error(y_test, predictions))
actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
submission = pd.DataFrame()

submission['Id'] = test.Id
feats = test.select_dtypes(

        include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
print ("Начальный предикшн: \n", predictions[:5], "\n")

print ("Финальный предикшн: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions

submission.head()
submission.to_csv('Final_ML_KOSSAY_A.csv', index=False)