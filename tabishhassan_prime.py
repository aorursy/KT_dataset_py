5# %% [code]

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
import sys

s=2

z=1

a=101#a=input('enter number')  

r=a

y=[2]

t=2

while(z<r) :

    x=0

    k=0

    while (k<z-1) :

        if s%t==0 :

           x=1

           break;

        else:

           x=0

           t=y[k]

           k=k+1

    if x==0:

     # print(z,s) 

      z=z+1

      y.append(s)

    s=s+1

#sizeof(y) 

print(z,s)



print("Memory size of  = "+str(sys.getsizeof(y))+ " bytes"  ) 

#please do vehury fast

tab=open("million_prime.txt",'w')

l1=str(y)

tab.write(l1)

tab.close()

print("finish")