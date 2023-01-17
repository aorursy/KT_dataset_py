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
2+1
6+6
9-4
number = 0

number
number = 43

number
number = 5

print(number)



# Add 2 to 5 and print the value of number

number = 5 + 2

print(number)





number
number = 5



if number >= 5:

    print("Number is greater than 5")

    

print("always printed")
amount = 15



text_info = "hello" * amount



print(text_info)
number_int = 5



type(number_int)
number_float = 19.666



type(number_float)

print(1+2)



print(8-3)



print (3*4)



#division

print(8/2)

print(8/3)



print(8//3)
8-3+2
-3 + 4 * 2
(-3 + 4) * 4
hat_height_cm = 25

my_height_cm = 206



total_height_meters = (hat_height_cm + my_height_cm) / 100



print("Height in meters = ", total_height_meters, "!")
# minimum

print(min(1,2,3))
#maximum

print(max(4,5,6))
print("minimum of 1,2,3 is ", min(1,2,3))
print(abs(32))

print(abs(-32))

print(float(10))

print(int(3.334))
print(int("345"))

print(int("3a4"))
print(int("345") + 3)

print("345" + 3)
# 5 / 2 1R

155643645 % 3
5 / 2
print(15 % 2)

print(14 % 2)
if 3 % 2 == 0:

    print("even")

else:

    print("odd")
print(7 % 3)

print(8 % 3)

print(9 % 3)

print(10 % 3)

print(11 % 3)

print(12 % 3)

for i in range(10):

    print(i)
for i in range(10):

    print(i+1)
for i in range(5,10):

    print(i+1)
for i in range(1,100, 3):

    print(i+1)
for number in range(100):

    print(number)
for i in range(21):

    if (i % 2 == 0): 

        print(i, " is even")

    else:

        print(i, " is odd")

print("end")
# importing the math library

import math as m
m.pi
number = 3.56

m.ceil(number)
m.floor(number)
m.sqrt(4)
m.sqrt(9)
m.pow(3,2)
m.pow(3,3)
# plot lib

import matplotlib.pyplot as plt
plt.plot([1,2,3,4])

plt.xlabel("numbers")

plt.ylabel("some numbers")
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.plot([1,2,3,4], [1,4,9,16], 'go')

plt.axis([0, 6, -10, 20])
np.arange(0., 5., 0.2)
t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs')

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
t1 = np.arange(0.0, 5.0, 0.1)

t2 = np.arange(0.0, 5.0, 0.02)



def f(t):

    return np.exp(-t) * np.cos(2*np.pi*t)



plt.figure(1)

plt.subplot(211) # 2, 1, 1 - numrows, numcols, fignum

plt.plot(t1, f(t1), 'bo')



plt.subplot(212) # 2, 1, 2 - numrows, numcols, fignum

plt.plot(t2, f(t2), 'k')
