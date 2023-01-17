# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
housing_data= pd.read_csv("../input/USA_Housing.csv")
housing_data.head()
housing_data.describe()
housing_data.info()
sns.pairplot(housing_data)
sns.distplot(housing_data["Price"])
x_indevalue=housing_data[["Avg. Area Income"
                              , "Avg. Area House Age","Avg. Area Number of Rooms","Avg. Area Number of Bedrooms", "Area Population"]]
y_depnvalue=housing_data["Price"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(x_indevalue,y_depnvalue,test_size=0.5,random_state=15)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
y_pred=lm.predict(X_test)
x_indevalue[1:2]
pred_test=lm.predict(x_indevalue[1:2])
pred_test
