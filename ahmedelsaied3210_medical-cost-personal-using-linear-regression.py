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

import matplotlib.pyplot as pl

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

data = pd.read_csv('../input/insurance/insurance.csv')

data.head()
data.corr()
data.isnull().sum()
data.corr()['charges'].sort_values()
X=data.drop(['charges'],axis=1)

y=data['charges']

X.head()
data['region'].value_counts()
labelencoder_x=LabelEncoder()

X['sex']=labelencoder_x.fit_transform(X['sex'])

X['smoker']=labelencoder_x.fit_transform(X['smoker'])

X['region']=labelencoder_x.fit_transform(X['region'])

X.head()

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=10)

rf.fit(x_train,y_train)
rf_predict=rf.predict(x_test)

mean_squared_error(y_test,rf_predict)