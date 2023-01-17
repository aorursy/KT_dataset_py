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
2+3

a=2

type(a)
a=2.6

type(a)
a='2.6'

type(a)
a="2.6"

type(a)
import keyword

print (keyword.kwlist)
a="data science"

a.capitalize()

a.lower()
a.upper()
a.isupper()
a.islower()
a.isalpha()
a.isalnum()
a.title()
a.isdigit()
a.split()
a="data science"

a.split('a')
a.split('s')
a.split('c')
a=23456
a-234.567
a="data science"
a.split('')
a.startswith('d')
a.startswith('D')
a.endswith('e')
a.endswith('s')
x=a.encode('utf 16')

x
a[0].upper()
a[0:3].upper()
a[1:5].lower()
a[-12:-9].upper()
a[99]
min(a)
max(a)
a.count('a')
a.count('a',1)
a.count('a',1,b)
a.index('a')
a.index('a',1)
a.index('a',1,b)
a.find('a')
a="    data   "
a.strip()
a.splitlines()
a.isspace()
a.lstrip()
a.rstrip()
a.isspace()
a="data science \n is awsome"
a.index("a")
a.strip()
a.replace("t","b")
a="data science learning is easy"

a="ence"in ""

a
a="data science learning is easy"

a="ence"not in ""

a
a="data"

b="science"

c=a+b

c

a="data"

b="science"

c=a+" "+b

c
a.find('a')

a.isspace()
a="\t"

a.replace("\t","xyz")
import math

math.ceil(5.6),math.floor(7.6)
math.floor(7.6)
math.sin(90),math.cos(90),math.tan(90)
math.asin(1),math.acos(1),math.atan(1)
math.sinh(1),math.cosh(1),math.tan(2.3)
math.e,math.expm1(10)
math.expm1(3)
math.log(10),math.log2(5),math.log10(10),math.exp(10)
pow(2,7),2**5
math.pow(4,5)
a=["machine","learning","training","awesome",2.6,40]

a,

type(a)
a[0].upper(),pow(a[-1],10)
a[1].lower()
a[2].upper()
pow(a[2],10)
pow(a[-2],10)
a=["data","science",["analytics","ml","deep learning"]]

a
len(a)
a[0],a[1],a[2]
a[-1]
a[-1][-1]
a[-1][-1].split()
a[-1][-1].split()[-1]
a[0]+a[-1][-1].split()[-1]
a=['data',["test",["abc","xyz"]]]
a[0],a[0][0]
a[0][0].upper(),a[1][1],a[0][1]
a[-1],a[-1][-1],a[-1][0]
a[-1][0].upper()
a[1][-1],a[1][-1][-1],a[1][-1][-1][-1]
a[0][-1],a[0][-1],a[1][-1]
import numpy as np

y=np.array(a)

type(y)
x=np.array([[1,2,3,4],[12,22,44,43],[13,24,34,56],[34,56,78,89]])

x
x.shape
x[:,:]
x[0:2,:]
x[0:2,2:]
x[:,[1,3]]
x[:,[1,-3]],x[0:2,:]
x[-3:-1,:],x[2:4,-3:-1],x[:,2:],x[2:,2:]
import numpy as np

x=np.random.randint(3,10)

x
x=np.random.randint(0,10,5)

x
x=np.random.randint(3,10,(4,3))

x
x.sum(axis=1)
x.sum(axis=0)
x.sum()
x.sum(),x.mean()
x.sum(axis=1),x.mean()
import numpy as np

A=np.matrix([[1,2,3,33],[4,5,6,66],[7,8,9,99]])

A
np.argmax(A),np.argmin(A)
np.argmin(A[:,:]),np.argmin(A[:1]),np.argmin(A[:,2]),np.argmin(A[1:,2])
np.argmin(A,axis=0)
np.argmax(A[:,:])
np.argmax(A[:1])
np.argmax(A[:,2])
np.argmax(A[1:,2])
np.argmin(A[:,:])
np.argmin(A,axis=0),np.argmax(A,axis=0)
np.argmin(A,axis=1),np.argmax(A,axis=1)
x=np.random.randint(3,10,(5,3))

x
print(x)
x.sum(),x.mean()
x.sum(axis=1),x.mean(axis=1)
x.sum(axis=1),x.mean(axis=1),x.min(axis=1)
x.sum(axis=0),x.mean(axis=0),x.min(axis=0)
x[0]
x[0]-x
(x[0]-x)**2
(((x[0]-x)**2).sum(axis=0))
import numpy as np

x=np.random.randint(0,10,(10,5))

x
np.add(x,3)
np.subtract(x,3)
np.multiply(x,2)
np.divide(x,2)
np.mod(x,2)
x=np.random.randint(0,10,(3,3))

y=np.random.randint(0,10,(3,3))

z=np.random.randint(0,10,(2,2))

x,y,z
np.add(x,y)
np.multiply(x,y)
x=np.random.randint(0,10,(5,3))

x
np.negative(x)
np.power(x,2)
np.abs(x)
np.floor_divide(x,2)
np.random.randint(0,10)
np.random.randint(0,10,10)
np.random.randint(0,10,(10,5))
np.random.randn(5,5)
np.zeros(5)
np.zeros((5,5))
np.ones(5)
np.ones((5,5))
np.arange(0,10)
np.linspace(0,10,100)
import numpy as np

x=np.array([["sowmya","maths"],["bhavani","physics"]])

x
np.char.lower(x)
np.char.upper(x)
import numpy as np

x=np.array([["sowmya","maths"],["10","20"]])

x
import numpy as np

x=np.random.randint(0,10,(10,3))

x
(x[7:8]-x[0:7]**2).sum(axis=1)
(x[8:9]-x[0:7]**2).sum(axis=1)
(x[9:10]-x[0:7]**2).sum(axis=1)
(x[7:8]-x[0:7]**2).sum(axis=1)+(x[8:9]-x[0:7]**2).sum(axis=1)+(x[9:10]-x[0:7]**2).sum(axis=1)
((x[7:8]-x[0:7]**2).sum(axis=1)+(x[8:9]-x[0:7]**2).sum(axis=1)+(x[9:10]-x[0:7]**2).sum(axis=1)).min()
((x[7:8]-x[0:7]**2).sum(axis=1)+(x[8:9]-x[0:7]**2).sum(axis=1)+(x[9:10]-x[0:7]**2).sum(axis=1)).argmin()
((x[7:8]-x[0:7]**2).sum(axis=1)+(x[8:9]-x[0:7]**2).sum(axis=1)+(x[9:10]-x[0:7]**2).sum(axis=1)).argmax()