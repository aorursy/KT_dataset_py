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
df=pd.read_csv('../input/z3-stattxt/Z3_Stat.txt', sep=" ", header=None)

df.columns = ['x1', 'x2','x3', 'x4', 'y', 'd'] 

df=df.drop('d',axis=1)

df
n=52

p=4
def ccor(x,y,n):

    return (n*np.array([x[i]*y[i] for i in range(n)]).sum()-x.sum()*y.sum())/np.sqrt((n*np.array([x[i]*x[i] for i in range (n)]).sum()-x.sum()*x.sum())*(n*np.array([y[i]*y[i] for i in range (n)]).sum()-y.sum()*y.sum()))
ccor(df['x1'],df['y'],52)
R=np.array(df.corr(method='pearson'))

df.corr(method='pearson')
# arr = np.random.normal(0,1,(4,4))
def alg_com_d(arr, i, j):

    return np.linalg.det(alg_com(arr,i,j))*((-1)**(i+j))
def alg_com(arr, i, j):

    return np.delete(np.delete(arr,i,axis=0), j, axis=1)
# print(arr)

# alg_com_d(arr, 3, 2)
Ryx=[alg_com_d(R,i ,4) for i in range(5)]

[alg_com_d(R,i ,4) for i in range(4)]
Rxx=[alg_com_d(R, i ,i) for i in range(4)]

[alg_com_d(R, i ,i) for i in range(4)]
Ryy=alg_com_d(R,4,4)
rxy=np.array([Ryx[i]/np.sqrt(Rxx[i]*Ryy) for i in range(4)])*(-1)

np.array([Ryx[i]/np.sqrt(Rxx[i]*Ryy) for i in range(4)])*(-1)
def tstat(r,n,k):

    return(np.abs(r)*np.sqrt((n-k-2)/(1-r**2)))  
tstat(rxy[3],52,2)#Погрешность уже в сотых
R2yx=1-np.linalg.det(R)/alg_com_d(R,4,4)

R2yx
np.sqrt(1-(1-R2yx)*(n-1)/(n-p-1))