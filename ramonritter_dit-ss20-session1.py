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



2+7

5+9
number = 5.3

number
if number >  5:

    print("number is greater than 5")



print("always printed")
hat_height_cm = 25

my_height_cm = 206

total_height_meters = (hat_height_cm + my_height_cm) / 100

print("Height in meters = ", total_height_meters, "!")
type(number)
print(min(1,2,3))

print(max(3,4,5))

print("minimum of 1,2,3 is", min(1,2,3))
print(abs(2-3))

print(float(30))

print(int(2.655))

print(int("234"))

type("3a4")



if 3 % 2 == 0:

    print("even")

else:

    print("odd")


for x in range(1, 11, 2):

    print(x)



for x in range(10):

    print(x+1)

    
x = 1

while x <= 20:

    if x % 2 == 0:

        print(x, "is even")

    else: 

        print(x, "is odd")

    x = x+1
for i in range(21):

    if i % 2 == 0:

        print(i, "is even")

    else: 

        print(i, "is odd")
# importing the math libary

import math as m

m
m.pi

print(m.ceil(m.pi))

print(m.floor(m.pi))
print(m.sqrt(16))

print(m.pow(3,3))
import matplotlib.pyplot as plt

plt.plot([1,2,3,4])

plt.xlabel("numbers")

plt.ylabel("some numbers")
plt.plot([1,2,3,4], [2, 4, 16, 1000], 'go')

plt.axis([0, 6, -10, 20])
y = np.arange(0., 5., 0.2)
plt.plot(y, y, 'r--', y, y**2, 'bs')
t1 = np.arange(0.0, 5.0, 0.1)

t2 = np.arange(0.0, 5.0, 0.02)



def f(t):

    return np.exp(-t) * np.cos(2*np.pi*t)



plt.figure(1)

plt.subplot(211) # 2, 1, 1 - numrows, numcols, fignum

plt.plot(t1,f(t1),'bo')



plt.subplot(212)

plt.plot(t2, f(t2), 'k')


plt.figure(1)

plt.subplot(211)

plt.plot([1,2,3])

plt.title('Easy as 1, 2, 3')
mu = 100

sigma = 15



x = mu + sigma * np.random.randn(10000)

plt.hist(x, normed=1, facecolor='g', alpha=0.75)



plt.xlabel("Smarts")

plt.ylabel('Probability')

plt.title('Histogram Test')

plt.text(120, 0.25, r'$\mu=100, \ \sigma=15$')

plt.grid(True)



plt.show()