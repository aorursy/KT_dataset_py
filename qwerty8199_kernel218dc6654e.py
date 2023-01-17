# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")

df = pd.DataFrame(data)
print(df.head)
x = df["YearsExperience"]

Y = df["Salary"]

a = x

b = Y

x.to_numpy()

Y.to_numpy()



#print(Y.describe())

#print(x,Y)

Y.values.reshape(-1,1)

x.values.reshape(-1,1)



x.head(5)

Y.head(5)

x = np.array(x)

x = [x]

Y = np.array(Y)

Y = [Y]

print(a.describe(),b.describe())
print(Y.head())
def return_cof(x,y,m,c):

    _c,_m = 0,0

    n = len(a)

    for i in range(1000):

        for j in range(n):

            #print(x[j],y[j])

            _m += (x[j]*(y[j]-m*x[j]+c))

            _c += (y[j]-(m*x[j]+c))  

        _m += -(2/n)*_m

        _c += -(2/n)*_c

        m += 0.001*_m

        c += 0.001*_c

        #print(m,c)

    print(m,c)

    return m,c
reg = LinearRegression()

reg.fit(x,Y)
m,c = 0,0

print("hii")

print(len(a))

for i in range(len(a)-2):

    m = m+((a[i+1]-a[i])/(b[i+1]-b[i]))

    m = m/2

    c = c+(b[i]-m*a[i])

    c = c/2

print("m,c ",m,c)

m,c = return_cof(a,b,m,c)
y_sk,y_my = [],[]

for i in range(len(a)):

    temp = (m*a[i])+c

    #print(a[i],temp)

    y_my.append(temp)

print(y_my)

y_sk = reg.predict(x)

plt.scatter(a,y_sk,color = "red")

plt.scatter(a,y_my,color = "blue")

#plt.scatter(x,Y,color="black")

plt.show()