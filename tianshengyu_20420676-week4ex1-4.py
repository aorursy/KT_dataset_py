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
x=int(input("miles_to_km_and_meters: "))

k=x/0.62137

m=k*1000

print(k,"km",m,"meters")
x=str(input("name: "))

y=int(input("age: "))

age=y+27

print("Hi",x,"!","In 2047 you will be",age,"!")
def odd(m):

    if m%2==0:

        return 0

    else:

        return m



print("three variables")

x=odd(int(input("x= ")))

y=odd(int(input("y= ")))

z=odd(int(input("z= ")))





if x > y and x > z:

    print("the largest odd number is",x)

elif y > x and y > z:

    print("the largest odd number is",y)

elif z > x and z > y:

    print("the largest odd number is",z)

elif x==y==z==0:

    print("three numbers are even")

else:

    print("three numbers are odd and equal")
numXs=int(input("How many times should I print the letter X?"))



while numXs > 0:

    print("x")

    numXs -=1