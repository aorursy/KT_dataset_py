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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data_path = '../input/hr_data.csv'

data = pd.read_csv(data_path)

data.head(15)
data['sales'].unique()
X_data =  data[['last_evaluation','number_project','average_montly_hours','time_spend_company']]

Y_data = data['satisfaction_level']
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test= train_test_split(X_data,Y_data, test_size= 0.3)
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
X_train.columns

print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
reg.intercept_
##Regression Formula

satisfaction_level = 0.62 + 0.25*last_evaluation - 0.039* number_project +0.000049*average_monthly_hours-0.014*time_spend_company
test_predicted = reg.predict(X_test)

test_predicted
data3 = X_test.copy()

data3['predicted_satisfaction_level']=test_predicted

data3['satisfaction_level']=y_test

data3.head()
error = y_test - test_predicted

mean_squared_error = error* error
error.abs().mean()
from sklearn.metrics import mean_squared_error,r2_score

## Mean squared error

print("Mean squared error: %.2f" % mean_squared_error(data3['satisfaction_level'],data3['predicted_satisfaction_level']))
np.sqrt(0.06)
## The determinant of coefficients aka r2

print(r2_score(y_test,test_predicted))
from sklearn.decomposition import PCA
pca = PCA(n_components=1)