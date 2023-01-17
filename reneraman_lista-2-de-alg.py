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
s=0

print("Escreva a tolerancia")

k = float(input())

x0=2

mult=1

s= s+mult*(1/x0)+1 #O 1 foi incluido diretamente na soma

x1= x0+2

E = abs((1/x0)-1)

i=1

while E >= k:

    x0=x0+2

    mult= mult*(-1)

    s= s+mult*(1/x0)

    x1= x0+2

    i=i+1

    E = abs(((1/x1))-(1/x0))

print("A soma é: ", s)
x0=-2

s= 1 + (1/x0)

i=1

x0= x0*(-1)+2

while x0 != 200:

    s= s + (1/x0)

    i=i+1

    if i%2 == 0:

        x0= (x0+2)*(-1)

    else:

        x0= (x0*(-1)+2)

   

print("A soma é: ", s)
print("Escreva um n")

n= int(input())

x0=1

x1=1

xi= x0+x1

i=3

x0=x1

x1=xi

while i!=n:

    xi= x0+x1

    i=i+1

    x0=x1

    x1=xi

print("O numero é: ",xi)
print("digite n")

n= int(input())

f0=1

i=1

fi= i * f0

f0=fi

while i != n:

    i=i+1

    fi= i * f0

    f0=fi    

print("O valor é: ",fi)
print("Informe o valor de x")

x= float(input())

c0=1-x

a0=1

c1= (c0)**2

a1=a0*(1+c0)

E=abs(a1-a0)

c0=c1

a0=a1

while E>=10**(-9):

    c1= (c0)**2

    a1=a0*(1+c0)

    E=abs(a1-a0)

    c0=c1

    a0=a1

print("O valor é: ", a1)
x2=1

x1=1

s=0

s=s + (x2/x1)

while x2 != 99 and x1 != 50:

    x2=x2+2

    x1=x1+1

    s=s + (x2/x1)

print("A soma é: ",s)