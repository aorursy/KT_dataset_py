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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
data=pd.read_csv('../input/board-games-prediction-data/games.csv')

data.head()
data.columns
data.shape
plt.hist(data['average_rating'])

plt.show
print(data[data['average_rating']==0].iloc[0])
data=data[data['users_rated']>0]

data=data.dropna(axis=0)

data.head()
data.shape
data.corr()
plt.hist(data['average_rating'])

plt.show
data.head()
X=data.drop(['id','name','type','average_rating'],axis=1)

y=data['average_rating']
X.head()
standardscaler=StandardScaler()

X=standardscaler.fit_transform(X)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
predict=lr.predict(x_test)  

from sklearn.metrics import mean_squared_error

mean_squared_error(predict,y_test)
from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor(random_state=7)

dt.fit(x_train,y_train)
dt_predict=dt.predict(x_test)

mean_squared_error(y_test,dt_predict)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=10)

rf.fit(x_train,y_train)
rf_predict=rf.predict(x_test)

mean_squared_error(y_test,rf_predict)