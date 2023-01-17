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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#load file
data=pd.read_csv("../input/Salary_Data.csv")
print(data.shape)
print(data.head())
#collecting x and y
X=data['YearsExperience'].values
Y=data['Salary'].values
#mean of x and y
mean_x=np.mean(X)
mean_y=np.mean(Y)
#total num of values
m=len(X)

#using the formula to calculate m and c in the line y=mx+c
numer=0
denom=0
for i in range(m):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
b1=numer/denom
b0=mean_y-(b1*mean_x)
#print coefficient
print(b1,b0)
#plotting values and regression line
max_x=np.max(X)
min_x=np.min(X)
#calculating line values x and y
x1=np.linspace(min_x,max_x,1000)
y1= b0+b1*x1
#plotting lines
plt.plot(x1,y1,color='#58b970',label='Regression line')
plt.scatter(X,Y,c='#ef5423',label='Scatter plot')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.legend()
plt.show()
#checking accuray using s2
ss_t=0
ss_r=0
for i in range(m):
    y_pred=b0+b1*X[i]
    ss_t+=(Y[i]-mean_y)**2
    ss_r += (Y[i] -y_pred) ** 2
r2=1-(ss_r/ss_t)
print(r2)
