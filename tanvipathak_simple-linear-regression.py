

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression as lm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")

df.head()
desc=df.describe()

print(desc)
sns.set()

sns.relplot(x="YearsExperience",y="Salary",data=df)
y=df.Salary

x=df.drop('Salary',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_test.head()


model=lm().fit(x_train,y_train)

predictions=model.predict(x_test)



plt.scatter(y_test,predictions)


c = [i for i in range (1,len(y_test)+1,1)]

plt.plot(c,y_test,color='r',linestyle='-')

plt.plot(c,predictions,color='b',linestyle='-')

plt.xlabel('Salary')

plt.ylabel('index')

plt.title('Prediction')

plt.show()