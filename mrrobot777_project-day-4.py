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

Iris = pd.read_csv("../input/iris/Iris.csv")

Iris.drop(["Species"], axis=1, inplace= True)

x=5.5

y=6.0

z=x-Iris['SepalLengthCm']

print(z)

w=y-Iris['SepalLengthCm']

v=x-Iris['SepalWidthCm']

u=y-Iris['SepalWidthCm']

t=x-Iris['PetalLengthCm']

s=y-Iris['PetalLengthCm']

r=x-Iris['PetalWidthCm']

q=y-Iris['PetalWidthCm']

df_col1=pd.concat([z,v,t,r],axis=1)

df_col2=pd.concat([w,u,s,q],axis=1)

print(df_col1,df_col2)