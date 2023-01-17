# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.isnull().sum()
corr=df.corr()

print(corr)
plt.scatter(df['reading score'],df['math score'],marker='o')

plt.xlabel('reading')

plt.ylabel('math')
plt.scatter(df['writing score'],df['math score'],marker='o')

plt.xlabel('write')

plt.ylabel('math')
plt.scatter(df['writing score'],df['reading score'],marker='o')

plt.xlabel('write')

plt.ylabel('read')
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(20,20))

ax=fig.add_subplot(111,projection='3d')

ax.scatter(df['writing score'],df['reading score'],df['math score'],c='r')



plt.show()
x=pd.DataFrame(np.c_[df['writing score'],df['reading score']],columns=['write','read'])

y=df['math score']

from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(x,y,test_size=0.3,random_state=5)
from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(X_train,Y_train)

pre=model.predict(X_test)

print(model.score(X_test,Y_test))
from sklearn.metrics import mean_squared_error

mse=mean_squared_error(Y_test,pre)

print(mse)

plt.scatter(Y_test,pre)
plt.plot(X_test,pre)

print(pre.shape)