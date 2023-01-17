# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        house_data = os.path.join(dirname, filename)

        print(house_data)

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
house_df = pd.read_csv(house_data,dtype='object')

house_df.head()
house_df.isnull().any()
Y = house_df.sqft_living

house_df.drop(['id','date','price'],axis = 1,inplace=True)

X = house_df.values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=100)

model = LinearRegression()

model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
r2_score(Y_test,Y_pred)
mean_squared_error(Y_test, Y_pred)
model.coef_
model.intercept_