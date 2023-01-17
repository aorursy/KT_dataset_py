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

nm = 'Perry'
nm
print("My last name is : ", nm)

name = ["Dwayne", "Kool", "Perry"]
print(name)

# In Python index starts at zero(0).
print("My first name is : ", name[0])
print("My last name is : ", name[2])

# Print my complete name.
for x in name:
  print(x)

max(5, 38, 56)

c = [3, 4, 8] # Represent a list
c
c.append(63)
c[1] # Python is 0 row index.

a = 10
if a > c[3]:
    print("a is more than index :", c[3])
elif a < c[3]:
    print("a is less than index :", c[3])
else:
    print("Error comparing : ", a, "and ", c[3])
    
if 4 in c:
    print("4 is a value in list c.")
else:
    print("4 does not exist in list c.")

if 'r' in nm:
     print("Your name is spell with  'r'")
else:
 print("Your name is not spell with  'r'")

if 'e' in nm:
     print("Your name is spell with: e")
else:
     print("Your name is not spell with: e")
     if 'p' not in nm:
         print("Your name is not spell with: p")
     else:
         print("Your name is not spell with: p")

# ............................ Practice ........................... #
# while loop. 
i = 0
while i <= a:
  print(i)
  i += 1

# A lambda function that adds 100 to the number passed in as an argument, and print the result.
add_func = lambda num : num + 100
print(add_func(5))

# A lambda function that adds 100 to the number (spare) passed in as an argument, and print the result.
add_mult_func = lambda num : num * num + 100
print(add_mult_func(5))

# Subtract both functions
print(add_mult_func(5) - add_func(5))


# Create a class with a variable equals five (5).
class ValueClass:
  val = 5.0

# Print and store class ValueClass into variable show_val 
show_val = print(ValueClass.val)

# Cast to float to integer.
output_val = int(454.8949) 
output_val

