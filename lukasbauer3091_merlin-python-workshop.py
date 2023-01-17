# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.
def maths(op, val1, val2):
    if op == "add":
        return val1 + val2
    elif op == "sub":
        return val1 - val2
    else:
        print("Please specify either \"add\" or \"sub\" when calling!")
        return 0
int1 = 44
int2 = 10
float1 = 2.1111
float2 = 81.988
str1 = "This is a string."
str2 = " Surprise, another string"
bool1 = True
bool2 = False

print(int1 + int2)
print(str1 + str2)
print("The first integer value is: %d" % int1)
print("The first string is: %s" % str1)
print("The first float value is: %.2f" % float1)

rmnd = int1%int2 # modulo -> what is the remainder when int1 is divided by int2
divFloat = float2/float1 # division with decimals
divIntSingle = int1/int2
divIntDouble = int1//int2
print(rmnd)
print(divFloat)
print(divIntSingle)
print(divIntDouble)
tu = (2, 10, "cat", 2.1, (1, 6, "dog"))
print(tu)
print(tu[4])
print(tu[4][1])
# tu[2] = "bigSnake"
# print(tu)

li = [10, 99, "magikarp", [2.222, "bedtime"]]
print(li)
print(li[-2])
li[3][0] = 25
print(li)
print(li[0:2])

#bool1 = 10 in li
#print(bool1)

li.append("merlin")
print(li)

li.insert(1, "5.5")
print(li)
if int2 < int1:
    print("int2 is smaller than int1")
elif int2 == int1:
    print("int2 equals int1")
else:
    print("Based on our superior deduction skills, int2 is bigger than int1")
while int2 < int1:
    print("Increasing int2 by 15")
    if int2 == 40:
        print("int2 equals the evil number! Exiting")
        break
    int2 += 15
for item in li:
    print(item)

for item in range(5):
    print(item)

for (num, item) in enumerate(li):
    print(num, item)
# simple dict

spyDict = {"name": "A. Powers", "position": "International Man of Mysteries"}
print(spyDict["name"])

spyDict["catchphrase"] = "Shagalicious baby!"
print(spyDict)
car = 'car'
ford = 'ford'
tesla = 'tesla'
cars = {} # new dictionary called cars
cars[tesla] = {} # within cars make a new key called tesla which is for another internal dictionary
cars[ford] = {} # repeat with ford
cars[tesla]['year'] = 2018 # within the tesla dictionary, make a key called "year" which has a value of 2018
cars[ford]['year'] = 2015
cars[tesla]['Type'] = 'Electric'
cars[ford]['Type'] = 'Diesel'
cars[ford]['Milage'] = 4850
cars[tesla]['Milage'] = 0

print(cars)

int1 = 10
int2 = 15
theSum = maths("add", int1, int2)
theDiff = maths("sub", int1, int2)
print(theSum)
print(theDiff)
try:
    result = 7/0
except ZeroDivisionError:
    print("Can't divide by zero!")
else:
    print("unknown error!")
finally:
    print("This is going to print always as it is in the finally - even if there is an error above!")
#classes!



