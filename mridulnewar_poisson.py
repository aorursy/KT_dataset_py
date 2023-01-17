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

train_df=pd.read_csv('../input/ds4train/ds4_train.csv')

test_df=pd.read_csv('../input/ds4-train/ds4_valid.csv')
x=train_df.iloc[0:,0:4]

x
li=[]

i=0

while(i<2500):

    li.append(1)

    i+=1
x.insert(0, "x0", li)

x
y=train_df.iloc[0:,4:]

y
x=x.to_numpy()

y=y.to_numpy()
def h(theta,x):

    return math.exp((theta.T).dot(x))
theta=np.zeros(5)

theta1=np.zeros(5)

theta.resize(5,1)

theta1.resize(5,1)

eps=1e-24

alpha=1e-12

sum=0

ans=0

a=0

for p in range(10):

    theta1=np.copy(theta)

    for j in range(5):

        sum=0

        for i in range(2500):

            sum+=alpha*(y[i]-h(theta1,x[i]))*x[i][j]

        theta[j]=theta1[j]+sum

    sum=0

    for i in range(5):

        sum+=(theta[i]-theta1[i])**2

    sum=math.sqrt(sum)

    if(sum<eps):

        print(p)

        break

theta

    
y
xtest=test_df.iloc[0:,0:4]

li=[]

i=0

while(i<250):

    li.append(1)

    i+=1
xtest
xtest.insert(0, "x0", li)

xtest
ytest=test_df.iloc[0:,4:]

ytest
xtest=xtest.to_numpy()

ytest=ytest.to_numpy()

for i in range (250):

    print(h(theta,xtest[i])-ytest[i])
ytest