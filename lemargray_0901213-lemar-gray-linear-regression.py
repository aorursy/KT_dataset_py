# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.DataFrame(np.random.randint(1,100,size=(100, 2)), columns=list('xy'))
df.head()
df.columns
plt.scatter(df['x'],df['y'])

plt.show()
X_train,X_test,Y_train,y_Test = train_test_split(df[['x']], df[['y']], test_size = 0.3)
#assign a variable to the LinearRegression function
reg = LinearRegression()
#pass xTrain & yTrain variables to the function to create the train/test model
reg.fit(X_train,Y_train)
reg.coef_
X_train.columns
print("Regression Coefficients")
pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
reg.intercept_
test_predicted = reg.predict(X_test)
test_predicted
print("Mean squared error: %.2f" % mean_squared_error(y_Test, test_predicted))
print('Variance score: %.2f' % r2_score(y_Test, test_predicted))

reg.score(X_test,y_Test)
