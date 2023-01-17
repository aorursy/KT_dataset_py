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
import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt 
train_df=pd.read_csv('/kaggle/input/ds5_train.csv')

valid_df=pd.read_csv('/kaggle/input/ds5_valid.csv')

test_df=pd.read_csv('/kaggle/input/ds5_test.csv')

X=train_df.iloc[0:,0:1]

m=300

li=[]

i=0

while(i<300):

    li.append(1)

    i+=1

X.insert(0, "x0", li)

y=train_df.iloc[0:,1:]

X=X.to_numpy()

Y=y.to_numpy()

w=np.arange(m*m)

w.resize(m,m)



x=valid_df.iloc[0:,0:1]

m=200

li=[]

i=0

while(i<200):

    li.append(1)

    i+=1

x.insert(0, "x0", li)

x

actual=valid_df.iloc[0:,1:]

x=x.to_numpy()

actual=actual.to_numpy()



def norm(xi,x):

    sum=0

    for i,j in xi,x:

        sum+=(i-j)**2

    return -1*sum
def fun(trex,point,t):

    return math.exp(norm(trex,point)/(2*t*t))
li=[]

w=np.zeros((300,300))

for test in range (200):

    for i in range (300):

        w[i][i]=fun(X[i],x[test],4)

    arr=(X.T.dot(w)).dot(X)

    theta=np.linalg.inv(arr).dot((X.T.dot(w)).dot(Y))

    li.append((theta.T).dot(x[test]))

li=np.array(li)

for i in range(200):

    print(li[i]-actual[i])
x1=[]

y1=[]

for i in range (200):

    x1.append(x[i][1])

for i in range (200):

    y1.append(actual[i])
plt.scatter(x1, y1, color= "green")  

plt.scatter(x1,li,color="blue")

# x-axis label 

plt.xlabel('x - axis') 

# frequency label 

plt.ylabel('y - axis') 

# plot title 

plt.title('') 

# showing legend 

plt.legend() 

  

# function to show the plot 

plt.show() 