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
import matplotlib.pyplot as plt

import seaborn as sns



#read data from dataset

dataset = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')



#print first five rows

dataset.head()

#print last five rows

dataset.tail()
#info about data

dataset.info()
# Remove multiple columns 

new_dataset = dataset.drop(['sl_no', 'gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'specialisation', 'salary'], axis = 1) 



#print first five data from new dataset

new_dataset.head()
# Replace 'workex' and 'status' column data to '0' and '1'



new_dataset['workex'].replace(to_replace ="Yes", 

                 value =1,inplace=True) 

new_dataset['workex'].replace(to_replace ="No", 

                 value =0,inplace=True) 



new_dataset['status'].replace(to_replace ="Placed", 

                 value =1,inplace=True) 

new_dataset['status'].replace(to_replace ="Not Placed", 

                 value =0,inplace=True) 



#print first five data

new_dataset.head()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(new_dataset[['ssc_p','hsc_p','degree_p','workex','etest_p','mba_p']],new_dataset.status,test_size=0.2)
X_test.head()
# Fitting Multiple Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()

regressor.fit(X_train, y_train)

# Predicting the Test set results

regressor.predict(X_test)

#Score of Multiple Logistic Regression model

regressor.score(X_train, y_train)
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)

rfr.fit(X_train, y_train)
#Score of Random Forest Regression model

rfr.score(X_train, y_train)
# Fitting SVR to the dataset

from sklearn.svm import SVR

regressor_svr = SVR(kernel = 'rbf')

regressor_svr.fit(X_train, y_train)
#Score of Support Vector Regression model

regressor_svr.score(X_train, y_train)
from sklearn.tree import DecisionTreeRegressor

regressor_dtr = DecisionTreeRegressor(random_state = 0)

regressor_dtr.fit(X_train, y_train)
#Score of Decision Tree Regression model

regressor_dtr.score(X_train, y_train)