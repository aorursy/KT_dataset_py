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
#Exercício 9

Tot = 0

a = 37

b = 38

c = 1

while (a != 1 or b != 2 or c != 37):

    Tot = Tot + a * b / c

    a = a - 1

    b = b - 1 

    c = c + 1

print(Tot)    
#Exercício 8 

Tot = 0

n = 1

x = 50

while (n != 50 or x != 1):

    Tot = Tot + 2**n / x

    n = n + 1 

    x = x - 1

print(Tot)    

    

#Exercício 7 

Tot = 0 

a = 1

b = 1

while (a != 99 or b != 50):

    Tot = Tot + a / b

    a = a + 2 

    b = b + 1

print(Tot)    
#Exercício 3

n = int(input())

if (n <= 2):

   print ( 'Erro n <= 2 ') 

else:

    Fib = 0

    ant = 1

    ant2 = 1

    x = 2

    print (1)

    print (1)

    while ( x != n ):

        Fib = ant + ant2

        print (Fib)

        ant2 = ant 

        ant = Fib

        x = x + 1

        

        

        

#Execício 4

n = int(input())

if (n <= 0):

    print ( 'Erro n <= 0')

else:

    Fat = 1

    x = 1

    while ( n != x ):

        Fat = Fat * n

        n = n - 1

    print (Fat)        
#Exercício 10

Tot = 0

a = 1

while (a != 10):

    if ( a % 2 == 0 ):

        Tot = Tot - a / a**2

    else:

        Tot = Tot + a / a**2

    a = a + 1

print(Tot)

#Exercício 11

Tot = 0

a = 1000

b = 1

while (b < 50):

    Tot = Tot + a / b * - 1

    a = a - 3

    b = b + 1

print ( Tot )    
#Exercício 13 

pi = 0;

precisao = 0.0001;

div = 1;

coef = 1;

signal = 1

while ( (div**2)** 1/2 >= precisao ):

    div = ( 4 / coef ) * control

    control = control * - 1

    pi = pi + div;

    coef = coef + 2;

print ( pi )    

    

    
#Exercício 14

pi = 0 

s = 0 

b = 1

signal = 1

x = 0

while ( x != 51 ):

    s += signal * (1 / (b**3))

    x += 1

    b += 2

    signal *= - 1

pi = (( s * 32 )**(1/3))

print ( pi )

    

    

    



#Exercício 15

x = int(input())

a = 25

b = 1

signal = 1

while ( b != 25 ):

    a += - 1

    

    

    
