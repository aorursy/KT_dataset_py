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
hyd=pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv")

hyd.head()

hyd.columns
hyd.groupby('year').agg("mean")["price"].plot(kind="bar")
hyd.year=hyd.year-2000
hyd.groupby('year').agg("mean")["price"].plot(kind="bar")
hyd.groupby('model').agg("mean")["price"]
hyd.groupby('fuelType').agg("mean")["price"].plot(kind="bar")
hyd.groupby('transmission').agg("mean")["price"].plot(kind="bar")
hyd.groupby('mileage').agg("mean")["price"].plot(kind="bar")
hyd.groupby('mpg').agg("mean")["price"].plot(kind="bar")
hyd.groupby('engineSize').agg("mean")["price"].plot(kind="bar")
hyd.groupby('tax(£)').agg("mean")["price"].plot(kind="bar")
hyd.drop(columns="model")
X2=pd.get_dummies(hyd)

X2
Y=hyd.price

X = X2.iloc[:,1:]

y = X2.iloc[:,2]





X.shape

y.shape

import sklearn.model_selection as model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65,test_size=0.35, random_state=101)

X_train

y
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))