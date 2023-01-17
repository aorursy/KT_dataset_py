# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv")
x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10,random_state=0)

regressor.fit(x_train,y_train)
plt.scatter(x,y,color='red')

x_grid=np.arange(min(x),max(x),0.1).reshape(-1,1)

plt.plot(x_grid,regressor.predict(x_grid),color='blue')

plt.title("Random Forest Regression on Salary-Year Data")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.plot()

from sklearn.metrics import r2_score

y_pred = regressor.predict(x_test)

r2_score(y_test,y_pred)