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
while True:
    try:      
        x = int(input("Enter a X Integer: "))
        y = int(input("Enter a Y Integer: "))
        z = int(input("Enter a Z Integer: "))
        break                               
    except ValueError:
        print("Please try again with integer number only!")

if x%2!=0 and y%2!=0 and z%2!=0:                     #if the remainder is not equal to 0, the number is odd
    print("The numbers of x,y and z are odd.")       #all numbers are odd, compare to find the largest
    if x>y and x>z: 
        print("%s is the largest odd number." % (x))
    elif y>x and y>z:
        print("%s is the largest odd number." % (y))
    elif z>x and z>y:
        print("%s is the largest odd number." % (z))

elif x%2!=0 and y%2!=0 and z%2==0:                   #two of them are odd
    print("The numbers of x and y are odd.")
    if x>y: 
        print("%s is the largest odd number." % (x))
    else:
        print("%s is the largest odd number." % (y))
elif x%2!=0 and z%2!=0 and y%2==0:
    print("The numbers of x and z are odd.")
    if x>z: 
        print("%s is the largest odd number." % (x))
    else:
        print("%s is the largest odd number." % (z))
elif y%2!=0 and z%2!=0 and x%2==0:
    print("The numbers of x and z are odd.")
    if y>z: 
        print("%s is the largest odd number." % (y))
    else:
        print("%s is the largest odd number." % (z))

elif x%2!=0 and y%2==0 and z%2==0:                   #only one of them is odd
    print("%s is the largest odd number." % (x))
elif y%2!=0 and x%2==0 and z%2==0:
    print("%s is the largest odd number." % (y))
elif z%2!=0 and x%2==0 and y%2==0:
    print("%s is the largest odd number." % (z))

else:                                                #None of them is odd
    print("The numbers of x,y and z are not odd.")
numXs = int(input('How many times should I print the letter X? '))
toPrint= ""
while numXs > 0:
    toPrint = toPrint + "X"
    numXs -=1
print(toPrint)