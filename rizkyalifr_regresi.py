# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualisasi data

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

import scipy



# Any results you write to the current directory are saved as output.
x=np.array((1,2,2.5,4,6,8,8.5))

y=np.array((0.4,0.7,0.8,1,1.2,1.3,1.4))
ax=plt.plot(x,y,'o')

plt.grid()

plt.xlabel('x')

plt.ylabel('y')
n = len(y) #jumlah data

xy=x*y #xy

xx=x**2 #xx
#Menghitung b

b=(n*xy.sum()-x.sum()*y.sum())/(n*xx.sum()-(x.sum())**2)

b
#Menghitung a

a=y.mean()-b*x.mean()

a
print("Persamaan regresi linearnya adalah : y = {:.4f}x + {:.4f}".format(b,a))
yreg=b*x+a #yregresi
dt=((y-y.mean())**2).sum()

d=((y-yreg)**2).sum()

r=np.sqrt((dt-d)/dt)
print("Nilai R adalah {:.4f} dan R^2 adalah {:.4f}".format(r,r**2))
plt.plot(x,y,'o') #scatter

plt.plot(x,yreg) #garis regresi

plt.grid()

plt.xlabel('x')

plt.ylabel('y')
#Menyatakan nilai logaritma dari data x dan y

xp=np.log10(x)

yp=np.log10(y)
n = len(y) #jumlah data

xy=xp*yp #xy

xx=xp**2 #xx
#Menghitung nilai b

bp=(n*xy.sum()-xp.sum()*yp.sum())/(n*xx.sum()-(xp.sum())**2)

bp
#menghitung nilai a

aa=yp.mean()-bp*xp.mean()

ap=10**aa

ap
print("Persamaan regresi pangkatnya adalah : y = {:.4f}x^{:.4f}".format(ap,bp))
yreg1=ap*(x**bp) #nilai yregresi
dt=((y-y.mean())**2).sum()

d=((y-yreg1)**2).sum()

r1=np.sqrt((dt-d)/dt)

print("Nilai R adalah {:.4f} dan R^2 adalah {:.4f}".format(r1,r1**2))
plt.plot(x,y,'o') #scatter

plt.plot(x,yreg1) #kurva regresi

plt.grid()

plt.xlabel('x')

plt.ylabel('y')
x=x

ye=np.log(y)
n = len(y) #jumlah data

xye=x*ye #xy

xx=x**2 #xx
#Menghitung b

b=(n*xye.sum()-x.sum()*ye.sum())/(n*xx.sum()-(x.sum())**2)

b
import math



#Menghitung a

A=ye.mean()-b*x.mean()

a=math.exp(A)

a
print("Persamaan regresi pangkatnya adalah : y = {:.4f}e^{:.4f}x".format(a,b))
yreg2=np.array(a*np.exp(b*x)) #yregresi
dt=((y-y.mean())**2).sum()

d=((y-yreg2)**2).sum()

r2=np.sqrt((dt-d)/dt)

print("Nilai R adalah {:.4f} dan R^2 adalah {:.4f}".format(r2,r2**2))
plt.plot(x,y,'o') #scatter

plt.plot(x,yreg2) #kurva regresi

plt.grid()

plt.xlabel('x')

plt.ylabel('y')
#Membuat Matriks A

A=np.zeros((4,4),dtype=float)

for i in range(0,4):

    for j in range(0,4):

        A[i][j]=(x**(j+i)).sum()

A       
#Membuat Matriks B

B=np.zeros((1,4), dtype=float)

for i in range(0,4) :

    B[0][i]=((x**i)*y).sum()

B
A=np.array([[7.00000000e+00, 3.20000000e+01, 1.99500000e+02, 1.43075000e+03],

       [3.20000000e+01, 1.99500000e+02, 1.43075000e+03, 1.09241250e+04],

       [1.99500000e+02, 1.43075000e+03, 1.09241250e+04, 8.60691875e+04],

       [1.43075000e+03, 1.09241250e+04, 8.60691875e+04, 6.90354656e+05]])

b=np.array([   6.8  ,   37.3  ,  251.75 , 1867.075])



Ab = np.hstack([A, b.reshape(-1, 1)])



n = len(b)



for i in range(n):

    a = Ab[i]



    for j in range(i + 1, n):

        b = Ab[j]

        m = a[i] / b[i]

        Ab[j] = a - m * b



for i in range(n - 1, -1, -1):

    Ab[i] = Ab[i] / Ab[i, i]

    a = Ab[i]



    for j in range(i - 1, -1, -1):

        b = Ab[j]

        m = a[i] / b[i]

        Ab[j] = a - m * b



X = Ab[:, 4]
X
a0=X[0]

a1=X[1]

a2=X[2]

a3=X[3]
print("Persamaan regresi pangkatnya adalah : y = {:.4f} + {:.4f}x + {:.4f}x^2 + {:.4f}x^3".format(a0,a1,a2,a3))
yreg3=a0+a1*x+a2*(x**2)+a3*(x**3)
dt=((y-y.mean())**2).sum()

d=((y-yreg3)**2).sum()

r3=np.sqrt(abs(dt-d)/dt)

print("Nilai R adalah {:.4f} dan R^2 adalah {:.4f}".format(r3,r3**2))

plt.plot(x,y,'o') #scatter

plt.plot(x,yreg3) #kurva regresi

plt.grid()

plt.xlabel('x')

plt.ylabel('y')
print("Keempat jenis regresi tersebut masing masing memiliki R^2 score :")

print("Linear            : {:.4f}".format(r**2))

print("Pangkat           : {:.4f}".format(r1**2))

print("Eksponen          : {:.4f}".format(r2**2))

print("Polinomial orde 3 : {:.4f}".format(r3**2))