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
import pandas as pd

data = pd.read_csv("../input/data.csv")
data.head() #First 5 rows
data.info() #information
x=data.iloc[:,:-1].values #Independent Variable also called feature matrix

y=data.iloc[:,1].values #dependent variable



#Y=mx+c, m slope, c intertcept
#Splitting the dataset in test and train

from sklearn.model_selection import train_test_split



#Machine Learning stuff will be in Sklearn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



#Dataset 100%=0.2==>20% for testing, remaining is to train..
import sklearn

print(sklearn.__version__) #For checking the version
from sklearn.linear_model import LinearRegression #Linear model Lib, LR is imported

from sklearn.ensemble import RandomForestRegressor
regressor= LinearRegression() #We are assigning the LR package to  regressors

regressorrandfor = RandomForestRegressor(n_estimators=11, random_state=0)
regressor.fit(x_train, y_train) #training data set is fitted

regressorrandfor.fit(x, y)
#predicting Test set results

y_pred=regressor.predict(x_test) #Prediticting Y=mx+c

y_pred

y_pred_rf= regressorrandfor.predict(x) 

y_pred_rf
import matplotlib.pyplot as plt #Visualizations
#Visualization--Training set

plt.scatter(x_train, y_train, color='red')

plt.plot(x_train, regressor.predict(x_train), color='green')

plt.title('Height V/S weight')

plt.xlabel('Weight')

plt.ylabel('Height')

plt.show()
#Visualizations--Test set

plt.scatter(x_test, y_test, color='red')

plt.plot(x_train, regressor.predict(x_train), color='green')

plt.title('Height V/S weight(testset)')

plt.xlabel('Weight')

plt.ylabel('Height')

plt.show()
import numpy as np
#Visualizations--Random Forest Regressors

X_grid=np.arange(min(x), max(x), 0.1)

X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(x, y, color='red')

plt.plot(x, regressorrandfor.predict(x), color='green')

plt.title('Turth of bluff (Random Forest Regressor)')

plt.xlabel('Weight')

plt.ylabel('Height')

plt.show()
regressorrandfor.score(x, y)
# print the coefficients

print(regressor.intercept_) #c

print(regressor.coef_) #m
import statsmodels.formula.api as smf #Statistical Analysis
lm1 = smf.ols(formula='Height ~ Weight', data=data).fit()
lm1.conf_int()
regressor.score(x, y) #this is from SklearnLib and Linear Regressor's out put
lm1.summary()