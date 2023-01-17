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
import pandas as pd

import numpy as np
data = pd.read_csv("../input/salary/Salary_Dataa.csv")
data
data.shape
data.info()
data.isnull().count()
X = data.iloc[:,:-1].values

Y = data.iloc[:,1].values
X
Y
import matplotlib.pyplot as plt
plt.scatter(X,Y,color = "blue")
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state = 0)

x_train
y_train
y_test
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)



y_pred = regressor.predict(x_test)
y_pred   
y_test   
residue = y_pred - y_test    # residue or error between actual and predicted salary

residue
plt.scatter(x_train,y_train,color="red")

plt.plot(x_train,regressor.predict(x_train),color="yellow")

plt.title("Salary VS Experience(Training Set)")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
plt.scatter(x_test,y_test,color="Red")

plt.plot(x_train,regressor.predict(x_train),color="yellow")

plt.title("Salary VS Experience(Training Set)")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
print(regressor.coef_)

print(regressor.intercept_)
y_test.shape
y_pred.shape
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



rmse = np.sqrt(mean_squared_error(y_test,y_pred))

r2 = r2_score(y_test,y_pred)                           #built-in function r2_score() indicates R-squared value 



print("RMSE =", rmse)

print("R2 Score=",r2)