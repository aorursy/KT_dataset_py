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

        

import matplotlib.pyplot as plt

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data= pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')
data.shape
data.info()
data.head()
x=data['YearsExperience'].values

y=data['Salary'].values

plt.scatter(x, y)

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
x_mean=np.mean(x)

y_mean=np.mean(y)
## calculation of line's m and b 



x_num = x-x_mean

y_num = y-y_mean

m=(sum(x_num*y_num))/(sum(x_num*x_num))
m
b=y_mean-(m*x_mean)
b
y1 = m*x+b

plt.plot(x, y1, '-r',label='Best Fit Line')

plt.scatter(x, y)

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
def fit(x,y):

    x_num = x-x_mean

    y_num = y-y_mean

    m=(sum(x_num*y_num))/(sum(x_num*x_num))

    b=y_mean-(m*x_mean)

    return m,b

    

def predict(val):

    m,b=fit(x,y)

    return (m*val)+b

    
fit(x,y)
predict(7.8)
class LinearRegression:

    def __init__(self):

        self.x_mean=0

        self.y_mean=0

        self.m=0

        self.b=0

    def fit (self,x,y):

        self.x=x

        self.y=y

        self.x_mean=np.mean(self.x)

        self.y_mean=np.mean(self.y)

        var1=(self.x-self.x_mean)*(self.y-self.y_mean)

        var2=(self.x-self.x_mean)**2

        self.m=(sum(var1)/sum(var2))

        self.b=self.y_mean-(self.m*self.x_mean)

    

    def predict(self,X):

        return (self.m*X+self.b)
model=LinearRegression()

model.fit(x,y)

model.predict(7.8)