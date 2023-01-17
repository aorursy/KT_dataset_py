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
cities = ["barcelona", "madrid", "istanbul", "paris"]
i = 0



for city in cities:

    print(i, city)

    i+=1
for i, city in enumerate(cities):

    print(i, city)
x_list = [1,2,3]

y_list = [2,4,6]
for i in range(len(x_list)):

    x = x_list[i]

    y = y_list[i]

    print(x, y)
for x,y in zip(x_list, y_list):

    print(x, y)
x = 10

y = -10
print("Before : x = %d, y = %d" %(x,y))
tmp = y

y = x

x = tmp
print("After : x = %d, y = %d" %(x,y))
print("Before : x = %d, y = %d" %(x,y))
x, y = y, x
print("After : x = %d, y = %d" %(x,y))
ages = {

    "Mary" : 31,

    "Jonathan" : 28

}
# age = ages["Dick"] this code has error, because dictionary has'nt "Dick" key.
if "Dick" in ages:

    age = ages["Dick"]

else: 

    age = "unknown"

    

print("Dick is %s years old."%age)
age = ages.get("Dick", "unknown")

print("Dick is %s years old."%age)
needle = "e"

haystack = ["a", "b", "c","d"]
found = False



for letter in haystack:

    if needle == letter:

        print("Found!")

        found = True

        break

        

if not found:

        print("Not found!")
for letter in haystack:

    if needle == letter:

        print("Found!")

        break

else: # if no break occured.

    print("Not found!")
f = open("/kaggle/input/text-simple/text.txt")

text = f.read()

for line in text.split("\n"):

    print(line)

    

f.close()
f = open("/kaggle/input/text-simple/text.txt")



for line in f:

    print(line)



f.close()
with open("/kaggle/input/text-simple/text.txt") as f:

    for line in f:

        print(line)
print("Converting!")

print(int("1"))

print("Done")
print("Converting!")



try:

    int("x")

except:

    print("Conversion failed!")

else:

    print("Conversion successful!")

finally:

    print("Done")