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

train_df=pd.read_csv('/kaggle/input/ds1-train/ds1_train.csv')

test_df=pd.read_csv('/kaggle/input/ds1-valid/ds1_valid.csv')

train_df
x=train_df.iloc[:,0:2]

y=train_df.iloc[:,2:3]

y
x=x.to_numpy()

y=y.to_numpy()

x
m=800

sum=0

for i in range (m):

    if(y[i]==1):

        sum+=1

phi=sum/m

phi
print(x[0])
temp=np.zeros((2,))

for i in range (m):

    if(y[i]==1):

        temp+=x[i].T

temp=temp/sum

mew1=temp

mew1    
temp=np.zeros((2,))

for i in range (m):

    if(y[i]==0):

        temp+=x[i].T

temp=temp/(m-sum)

mew0=temp



mew0   
mew0=mew0.reshape(2,1)

mew1=mew1.reshape(2,1)

sigma=np.zeros((2,2))

for i in range (m):

    if(y[i]==1):

        sigma+=(x[i].reshape((1,2)).T-mew1).dot((x[i].reshape((1,2)).T-mew1).T)

    else:

        sigma+=(x[i].reshape((1,2)).T-mew0).dot((x[i].reshape((1,2)).T-mew0).T)

sigma=sigma/m

sigma


theta=(np.linalg.inv(sigma)).dot(mew1-mew0)

theta
theta0=-1*np.log((1-phi)/phi)+0.5*((mew0.T.dot(np.linalg.inv(sigma))).dot(mew0)-mew1.T.dot(np.linalg.inv(sigma)).dot(mew1))

theta0

mew0.T.dot(np.linalg.inv(sigma)).dot(mew0)

print(mew1.T.dot(np.linalg.inv(sigma)).dot(mew1))
test_df
xtest=test_df.iloc[0:,0:2]

xtest

xtest.to_numpy()
li=[]

for i in range (len(xtest)):

    temp=1/(1+math.exp(-1*(theta.T.dot(x[i])+theta0)))

    if(temp<0.5):

        li.append(0)

    else:

        li.append(1)

li
ytest=test_df.iloc[0:,2:3]

ytest.to_numpy()
li=np.array(li)

li
c=0

for i in range (100):

    if(li[i]!=ytest[i]):

        c+=1

print(c)