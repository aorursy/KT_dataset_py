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
salary=pd.read_csv("/kaggle/input/salary/Salary.csv")
import matplotlib.pyplot as plt

import seaborn as sns
salary.head(2)
salary.Salary.sum()
salary.describe
salary.info()
salary.isnull().sum()
plt.figure(figsize=(20,5))

sns.set(style="darkgrid")

a=sns.countplot(x="Salary",data=salary)

a.set_xticklabels(a.get_xticklabels(),rotation=75)

plt.show()
plt.figure(figsize=(20,5))

sns.distplot(salary["Salary"],color="blue")
plt.figure(figsize=(20,10))

sns.jointplot(x="YearsExperience",y="Salary",data=salary)
plt.figure(figsize=(20,10))

sns.jointplot(x="YearsExperience",y="Salary",data=salary,kind="reg",color="red")
plt.figure(figsize=(20,10))

sns.jointplot(x="YearsExperience",y="Salary",data=salary,kind="hex")
sns.heatmap(salary.corr(),annot=True)
plt.figure(figsize=(20,5))

sns.distplot(salary["YearsExperience"],color="red")
sns.pairplot(salary)
X=salary[["YearsExperience"]]
y=salary["Salary"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
pre=lm.predict(X_test)
plt.figure(figsize=(20,5))

sns.distplot(y_test-pre,color="green")
from sklearn.metrics import mean_squared_error
mean_squared_error(pre,y_test)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(pre,y_test)
a=plt.scatter(X_train,y_train)
