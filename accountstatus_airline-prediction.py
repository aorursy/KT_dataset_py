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
import plotly.express as px

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
data=pd.read_csv('../input/air-india-passenger-prediction/Air India Passenger.csv')
data.head()
data['Month']=pd.to_datetime(data['Month'])
px.line(data,'Month','#Passengers')
data['Month']=data['Month'].map(pd.datetime.toordinal)
lr=LinearRegression()
X=data['Month'].values

y=data['#Passengers']
X=X.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
lr.fit(X_train,y_train)
y_hat=lr.predict(X_test)
print('The root mean squared error of using the Linear Regression is :',np.sqrt(mean_squared_error(y_hat,y_test)))