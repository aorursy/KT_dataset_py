# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
data = pd.read_csv('../input/housing.csv')
data.head()
data.shape
data.describe()
data.count()
data['total_bedrooms'].fillna(data['total_bedrooms'].mean(), inplace=True)
data.count()
data.describe()
data.columns
data['ocean_proximity'].value_counts()
data['ocean_proximity'] = data['ocean_proximity'].map({'<1H OCEAN':'1','INLAND':2,'NEAR OCEAN':3,'NEAR BAY':4,'ISLAND':5})
X_columns = ['longitude', 'latitude', 'total_rooms', 'total_bedrooms',
       'population', 'median_income','housing_median_age', 'households']
X_train, X_test, y_train, y_test = train_test_split(data[X_columns], data['median_house_value'], random_state=42)
data.tail()
linReg = LinearRegression()
#data['ocean_proximity'][0]
linReg.fit(X_train, y_train)
y_pred_train = linReg.predict(X_train)
y_pred_test = linReg.predict(X_test)
r2_score(y_test, y_pred_test)
r2_score(y_train,y_pred_train)
