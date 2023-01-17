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
data=pd.read_csv('../input/50-startups/50_Startups.csv')

data.head()
data=pd.get_dummies(data)
data.isnull().sum()
X=data.drop(['Profit'],axis=1)

y=data['Profit']
X.head()
print(y.head())
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
sc_X=StandardScaler()

x_train=sc_X.fit_transform(x_train)

pd.DataFrame(x_train).head()
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
predict=lr.predict(x_test)  
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,predict)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=10)

rf.fit(x_train,y_train)
rf_predict=rf.predict(x_test)

mean_squared_error(y_test,rf_predict)