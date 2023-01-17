# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/kc_house_data.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/kc_house_data.csv')

inspection = dataset.head()
all_params=list(dataset)

import seaborn as sns

sns.pairplot(dataset,

              x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'waterfront','condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built','sqft_living15', 'sqft_lot15','yr_renovated'],

              y_vars=["price"],

              size =30,

              kind ="reg"

              )
x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'waterfront','condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built','sqft_living15', 'sqft_lot15','yr_renovated']

print(list(set(all_params) - set(x_vars)))
y = dataset.iloc[:, 2].values

dataset = dataset.drop(['date','price','lat', 'zipcode', 'view', 'id', 'long'], axis =1)

X = dataset.iloc[:, ].values

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
accuracy = regressor.score(X_test, y_test)

print("Accuracy: {}%".format(int(round(accuracy * 100))))
