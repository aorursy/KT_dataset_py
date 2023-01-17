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



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv", sep=",")
data.head()
data.info()
data.describe()
data.isna().sum()
sns.pairplot(data)
ocean = pd.factorize(data.ocean_proximity)
ocean
data['ocean'] = pd.factorize(data.ocean_proximity)[0]
data.head()
sns.pairplot(data.iloc[:,3:])
a = pd.Series(data.median_income)

b = pd.Series(data.households / data.population)

c = pd.Series(data.total_rooms / data.households)

d = pd.Series(data.total_bedrooms / data.households)

plt.scatter(x=c,y=d)
e = pd.get_dummies(data.ocean_proximity)
X = pd.DataFrame()

X['median_income'] = a

X['population_density'] = b

X['housing_density'] = c

X = pd.concat([X,e],axis=1)

X.head()
y = np.array(data.median_house_value).reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

scaled_X_train = scaler.transform(X_train)

scaled_X_test = scaler.transform(X_test)

scaler_out = StandardScaler()

scaler_out.fit(y_train)

scaled_y_train = scaler_out.transform(y_train)

scaled_y_test = scaler_out.transform(y_test)
from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(scaled_X_train,scaled_y_train)

clf.score(scaled_X_test,scaled_y_test)

pred = scaler_out.inverse_transform(clf.predict(scaled_X_test))

plt.scatter(pred,y_test)
err = pred - y_test

plt.hist(err,bins=50)
((err * err).mean()) ** (0.5)
from sklearn import tree

clfTree = tree.DecisionTreeRegressor()

clfTree.fit(scaled_X_train,scaled_y_train)

pred_tree = scaler_out.inverse_transform(clfTree.predict(scaled_X_test))

plt.scatter(x=pred_tree,y=y_test)
err_tree = pred_tree.reshape(-1,1) - y_test

plt.hist(err_tree,bins=50)
((err_tree * err_tree).mean()) ** (0.5)