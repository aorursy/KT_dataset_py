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
df=pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
print(df)
len(df)
len(df.columns)
df.shape
for i in df.columns:

    print(i)
df.Surname
df.head()

df.iloc[0]
df.loc[[0,1],['CustomerId','Surname']]
df['CreditScore'].mean()
df.dropna()
import matplotlib.pyplot as plt
%matplotlib inline
df['Age']
df['EstimatedSalary']

plt.plot(df['Age'],df['EstimatedSalary'])

plt.xlabel("AGE")

plt.ylabel("EstimatedSalary")
x=df.loc[[0,1,2,3,4],['Age']]


print(x)
y=df.loc[[0,1,2,3,4],['EstimatedSalary']]
y
y1=y['EstimatedSalary']
x1=x['Age']
plt.plot(x1,y1)

plt.xlabel('AGE')

plt.ylabel('Estimated_Salary')
y=df.loc[[0,1,2,3,4],['Age','EstimatedSalary']]
y.plot.bar()
y=df.loc[[0,1,2,3,4],['Age']]
y.plot.bar()
y=df.loc[[0,1,2,3,4],['Age','EstimatedSalary']]

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.plot(y['Age'],y['EstimatedSalary'])

import seaborn as sns
df['Balance']
sns.distplot(df['Balance'])
sns.distplot(df['Balance'],kde=False,bins=30)