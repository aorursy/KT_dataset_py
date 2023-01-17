# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np# linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import array
from scipy.linalg import svd
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
a= array([[4,0],[3,-5]])
m=2
n=2
print(a)
u,s,v=svd(a)

print(u)
print(s)
print(v)
at=a.T
A=at.dot(a)
#print(A)

w,v =np.linalg.eig(A)
#print(w)
#print(v)

s = np.sqrt(w)
S=np.zeros((n,n))
for i in range(n):
    S[i][i]=s[i]

Sinv = np.linalg.inv(S)
print(S)

Vt =v.T
print(Vt)


U=a.dot((v.dot(Sinv)))
print(U)


Ans= U.dot((S.dot(Vt)))
print(Ans)