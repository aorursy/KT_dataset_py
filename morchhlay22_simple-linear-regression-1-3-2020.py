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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sal = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")
sal.head()
sal.info()
sal.isnull().sum()
sal[sal['Salary']==sal['Salary'].max()]
sns.set_style('whitegrid')

sns.FacetGrid(sal,height=6).map(plt.scatter,'YearsExperience','Salary').add_legend()

plt.show()
plt.figure(figsize=[10,10])

sns.barplot(x="YearsExperience",y="Salary",data=sal)
plt.figure(figsize=[10,10])

sns.countplot(x="YearsExperience",data=sal)
from sklearn.model_selection import train_test_split
X = sal[['YearsExperience']]

y = sal['Salary']
X_train , X_test, y_train , y_test =train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)
data_output=pd.DataFrame({"actual":y_test,"predicted":y_predict})
data_output
plt.scatter(X_train,y_train,color="red")

plt.plot(X_train,lr.predict(X_train),color="blue")

plt.title("salary vs experience")

plt.xlabel("experience")

plt.ylabel("salary")

plt.show()
plt.scatter(X_test,y_test,color="red")

plt.plot(X_train,lr.predict(X_train),color="blue")

plt.title("salary vs experience(x_test)")

plt.xlabel("experience")

plt.ylabel("salary")

plt.show()