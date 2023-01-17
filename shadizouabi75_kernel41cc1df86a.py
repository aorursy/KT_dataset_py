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
%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression # I didn't use function
X= np.array([1,10,20,40,50,70,80,90,120])

Y= np.array([3,20,90,110,130,170,150,200,260])
mean_x=np.mean(X)   # calculate total / size of array

mean_y=np.mean(Y)
n=len(X)
print (n)
numer=0   # independent numbers which is x values

denom=0   # dependent numbers which is y values
for i in range(n):

    numer+=(X[i]-mean_x)*(Y[i]-mean_y)

    denom+=(X[i]-mean_x)**2

    b1=numer/denom   # slope

    b0=mean_y-(b1*mean_x) # y intercet

    
print (b1,b0)   # coefficient
# plotting values and regression line

max_x=np.max(X)+5

min_x=np.min(X)-5
print(max_x)
print(min_x)
#calculating line values x and y

x= np.linspace(min_x,max_x,30)

y=b0+b1*x
print(y)

print(x)
print(x)
plt.plot(x,y,color='#59b970',label='Regression line')

plt.scatter(X,Y,color='#ef5423',label='Scatter plot')

plt.xlabel('X values')

plt.ylabel('Y values')

plt.legend()

plt.show