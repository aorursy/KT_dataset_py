# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
data.head()
data.dtypes
plt.figure(figsize = (17,7))

plt.subplot(1,2,1)

sns.countplot(x = data['city'])

plt.title('number of diffrent citis in dataset')

plt.subplot(1,2,2)

sns.countplot(x = data['rooms'])

plt.title('number of diffrent rooms in dataset')

plt.show()
rio = data[ data['city'] == 'Rio de Janeiro']

porto = data[data['city'] == 'Porto Alegre']

plt.figure(figsize = (17,7))

plt.subplot(1,2,1)

sns.countplot(x = porto['animal'])

plt.subplot(1,2,2)

sns.countplot(x = rio['animal'])

plt.show()
data = data.drop('total (R$)',axis = 1)
data ['animal'] = data['animal'].replace('acept', 1)

data ['animal'] = data['animal'].replace('not acept', 0)

data['furniture'] = data['furniture'].replace('furnished' , 1)

data['furniture'] = data['furniture'].replace('not furnished' , 0)
data['floor'] = data['floor'].replace('-', np.nan)

data.dropna(inplace = True)
x = data.drop('rent amount (R$)' , axis = 1)

y = data['rent amount (R$)']

x = x.values

y = y.values
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype= np.float64 )
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.2)
regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

print("regressor score is", regressor.score(x_train , y_train))

rscore = r2_score(y_test,y_pred)

print("r2score is", rscore)

mae = mean_absolute_error(y_test,y_pred)

print("mean absolute error is",mae)

mse = mean_squared_error(y_test,y_pred)

print("mean squared error is",np.sqrt(mse))