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
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
data.head()
data.corr()
data = data.drop(columns=['Serial No.'])
data.info()
data.columns
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']]

y = data['Chance of Admit ']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

model.score(X_test,y_test)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)

regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train,y_train)
regr.score(X_test,y_test)