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
from numpy import*

import matplotlib.pyplot as plt



x=[1,2,4,3,5]

y=[1,3,3,2,5]



mx=mean(x)

my=mean(y)

dx2=0

dxy=0

for i in range(len(x)):

    dxy+=(x[i]-mx)*(y[i]-my)

    dx2+=(x[i]-mx)**2

   

m=dxy/dx2



c=my-m*mx



print("The Regression line : "+"y="+str(m)+"x + ",c)



yp=[0]*len(x)

for i in range(len(x)):

    yp[i]=m*x[i]+c

   

   

plt.plot(x,y,'bo')

plt.plot(x,yp,'-g')

plt.show()



# Calculating Root Mean Square Error

d2=0



for i in range(len(x)):

    d2+=(y[i]-yp[i])**2

   

rmse=(d2/len(x))**0.5    

print("Root Mean Square Error:",rmse)  
