# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

a=np.random.random_integers(18,45,100)

a
a_mean=a.mean()

a_mean

a_median=np.median(a)

a_median
minn=np.amin(a)

maxx=np.amax(a)

razmah=maxx-minn

razmah
a2=a[a>=30]

a3=a2.size/a.size

a3
a4=a[a<=20]

a5=a4.size/a.size

a5
b=44

c=np.in1d(b,a).all()#функция для проверки есть ли б в а

c
a_std=a.std()

z=a_mean-a_std

z1=a_mean+a_std

d1=a[(a>=18)&(a<z)]

d2=a[(a>=z)&(a<=z1)]

d3=a[(a>z1)&(a<=45)]

s1=np.sort(d1)

s2=np.sort(d2)

s3=np.sort(d3)

s1[:]=1

s2[:]=2

s3[:]=3

s1



d=np.hstack((s1, s2,s3))

d

z = np.random.randint(18, 45, 10)

d1=np.hstack((d,z))

d1