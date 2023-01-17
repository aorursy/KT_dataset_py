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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn import linear_model
cereal = pd.read_csv('/kaggle/input/80-cereals/cereal.csv')
cereal.describe()
train = pd.DataFrame(cereal[0:70])

test = pd.DataFrame(cereal[70:])
train.dtypes
train.type.value_counts()
train.columns
plt.scatter(train.sugars, train.rating)

plt.xlabel('Sugar')

plt.ylabel('Rating')

plt.title('Rating vs. Sugar content in cereal')
plt.scatter(train.calories, train.rating)

plt.xlabel('Calories')

plt.ylabel('Rating')

plt.title('Rating vs. Calories content in cereal')
plt.scatter(train.fiber, train.rating)

plt.xlabel('fiber')

plt.ylabel('Rating')

plt.title('Rating vs. fiber content in cereal')
plt.scatter(train.protein, train.rating)

plt.xlabel('protein')

plt.ylabel('Rating')

plt.title('Rating vs. protein content in cereal')
plt.scatter(train.sugars, train.calories)

plt.xlabel('Sugar')

plt.ylabel('Calories')

plt.title('Calories vs. Sugar content in cereal')
plt.scatter(train.sugars, train.fiber)

plt.xlabel('Sugar')

plt.ylabel('fiber')

plt.title('fiber vs. Sugar content in cereal')
model = linear_model.LinearRegression()



X = pd.DataFrame(train.sugars)

y = train.rating



model.fit(X = X, y = y)



print('R2 = ',model.score(X = X, y = y))

print('coeff = ',model.coef_)

print('intercept = ',model.intercept_)
plt.scatter(train.sugars, train.rating)

plt.xlabel('Sugar')

plt.ylabel('Rating')

plt.title('Rating vs. Sugar content in cereal')



plt.plot(np.arange(0,15,0.1),model.predict(pd.DataFrame(np.arange(0,15,0.1))), color = 'red')
model1 = linear_model.LinearRegression()



X = pd.DataFrame([train.sugars, train.fiber]).T

y = train.rating



model1.fit(X = X, y = y)

print('R2 = ',model1.score(X = X, y = y))

print('coeff = ',model1.coef_)

print('intercept = ',model1.intercept_)
y_pred = model1.predict(X)
rmse = mean_squared_error(y_true = y, y_pred = y_pred)**0.5

rmse
idx = train[train['fiber']>8].index

idx
train.drop(idx, axis = 0, inplace = True)
model2 = linear_model.LinearRegression()



X = pd.DataFrame([train.sugars, train.fiber]).T

y = train.rating



model2.fit(X = X, y = y)

print('R2 = ',model2.score(X = X, y = y))

print('coeff = ',model2.coef_)

print('intercept = ',model2.intercept_)
y_pred = model2.predict(X)

rmse = mean_squared_error(y_true = y, y_pred = y_pred)**0.5

rmse
predicted_y = model2.predict(pd.DataFrame([test.sugars,test.fiber]).T)

predicted_y
test