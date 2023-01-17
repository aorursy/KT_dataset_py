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
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import metrics
data = pd.read_csv("../input/student_scores.csv")
data.describe()
data.head()
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 
X_train,X_test,y_train,y_test=train_test_split(X,y, shuffle=False)
regressor= LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
print('Mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('Accuracy:',metrics.r2_score(y_test,y_pred))
plt.scatter(X_train,y_train,color='red')
plt.title(" train data set ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#for ploting the data  for test data with predicted data 
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title(" (Test data set)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()