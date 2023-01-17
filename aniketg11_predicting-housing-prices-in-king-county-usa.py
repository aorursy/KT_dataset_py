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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
df = pd.read_csv('../input/kc_house_data.csv')
df.head()

Y = df['price']
X = df.drop(['price'], axis=1)
X.head()
Y.head()
Y = Y.to_frame()
type(Y)
Y.head()
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10,6)

df['price'].describe()
X_train['price'] = y_train
X_test['price'] = y_test
X_train.head()
X_test.head()
print(X_train.price.skew())
plt.hist(X_train.price, color='blue')
plt.show()
target = np.log(X_train.price)
print('Skew is ', target.skew())
plt.hist(target, color='blue')
plt.show()
numeric_features = X_train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)
corr = numeric_features.corr()
corr['price'].sort_values(ascending=False)
corr['price'].sort_values(ascending=False)[-5:]
#to generate scatter plots and visualize relationships betweem sqft_living,grade,sqft_above,sqft_living15 and price
plt.scatter(x=X_train['sqft_living'], y=target)
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()
plt.scatter(x=X_train['grade'], y=target)
plt.xlabel('grade')
plt.ylabel('price')
plt.show()
plt.scatter(x=X_train['sqft_above'], y=target)
plt.xlabel('sqft_above')
plt.ylabel('price')
plt.show()
plt.scatter(x=X_train['sqft_living15'], y=target)
plt.xlabel('sqft_living15')
plt.ylabel('price')
plt.show()
nulls = pd.DataFrame(X_train.isnull().sum().sort_values(ascending=False)[:25])
nulls
categorical = X_train.select_dtypes(exclude= [np.number])
print(categorical.describe())
X_train = X_train.drop(['id', 'price', 'date'], axis=1)
X_train.head()

data = X_train.select_dtypes(include=np.number)
data.head()


#X_test
test_target = np.log(y_test.price)
X_test.head()
y_test.head()
X_test.head()
X_train.head()
X_test = X_test.drop(['id','date','price'], axis=1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


y = y_train
x = X_train
lr = linear_model.LinearRegression()
model = lr.fit(x, y)
print('R^2 =', model.score(X_test, y_test.price))
predictions = model.predict(X_test)
print('RMSE ', mean_squared_error(y_test,predictions ))
plt.scatter(x=predictions, y=y_test, alpha=0.75)
plt.show()
for i in range(-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(x, y)
    pred_ridge = ridge_model.predict(X_test)
    plt.scatter(pred_ridge, y_test, alpha=0.75, color='b')
    plt.xlabel('Predicted Price')
    plt.xlabel('Actual price Price')
    plt.show()