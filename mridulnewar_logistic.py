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

train_df=pd.read_csv('../input/ds1-train/ds1_train.csv')

test_df=pd.read_csv('../input/ds1-valid/ds1_valid.csv')
x=train_df.iloc[0:,0:2]

m=800
li=[]

i=0

while(i<800):

    li.append(1)

    i+=1
x.insert(0, "x0", li)

y=train_df.iloc[0:,2:]
x=x.to_numpy()
y=y.to_numpy()
def g(theta,x):

     return 1/(1+math.exp(-1*((theta.T).dot(x.T))))

     

    
def gradient(theta):

    li=[]

    i=0

    while(j<n):

        sum=0

        while(i<m):

            sum+=y[i]-g(theta,x[i])*x[i][j]

            i+=1

        li.append((sum*-1)/m)

        j+=1

        i=0

    li= np.array(li) 

    li=li.T

    return li

    

    

    
def hessian(theta):

    rows, cols = (3, 3) 

    arr = [[0]*cols]*rows

    i=0

    j=0

    k=0

    while(j<3):

        while(k<3):

            sum=0

            while(i<m):

                sum+=(x[i][j]*x[i][k]*g(theta,x[i])*(1-g(theta,x[i])))

                i+=1

               # print(sum)

            arr[j][k]=sum/m

            k+=1

            i=0

        j+=1

        k=0

    arr = np.array(arr)

    print(arr)

    return np.linalg.inv(arr) 

    
theta=np.zeros(3)

theta.resize(3,1)

ans=0

while(1):

    i=0

    theta1=theta-(hessian(theta).dot(gradient(theta)))

    while(i<3):

        ans+=theta[i]-theta1[i]**2

        i+=1

    if(ans<1e-10):

        break

    else:

        theta=theta1

    ans=0

theta

    