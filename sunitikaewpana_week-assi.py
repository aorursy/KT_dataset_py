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
2+3
print ("Hello World")
name = "Plenoi"
print("Hello "+name+" !")
country = input("Where are you from = ")
print("I am from " + country)
age = input("What is your age = ")
print("I am "+age+" !!")
# Ex. 1, How to fix this syntax error
tenyears_age = int(age) + 10
print(tenyears_age)
age = 25
print(age)
goal = "Being Data Scientist !!"
print(goal)
pi = 3.14
print(pi)
#Ex. 2, How to fix this syntax error
print("The pi is equal to" + str(pi))
5 + 2 - (4*3)
5 // 2
5 / 2
5 % 2
5 ** 2
a = input("What is your A = ")
b = input("What is your B = ")
# Ex. 3, How to fix this runtime error
c = int(a) + int(b)
print(c)
greeting = "Hello"
name = ' World'
print(greeting + name)
print("Hello 'World' PyThon")
print('Hello "World" PyThon')
msg = "Hello " + "World!"
print(msg)
"ap" in "apple"
"z" in "apple"
name = "Plenoi"
age = 20
msg = f"Hello my name is {name} and my age is {age}"
print(msg)
print("Hello my name is "+name+" and my age is "+str(age))
print("Hello my name is %s and my age is %.2f" % (name,age))
my_string = "Hello World"
my_string.upper()
my_string.lower()
my_string.find("World")
my_string.index("World")
my_string.capitalize()
my_string.replace("World","")
my_string.rstrip()
my_string[2]
my_string.join("ABCD")
# Ex.4
my_string.split(" ")
%reset
L = [1,2,3,"Hello","World"]
L
L.append("Plenoi")
L
L.remove(3)
L
L.pop(3)
L
L = [[1,2,3],
     [4,5,6],
     [7,8,9]]
L
L.reverse()
L
L.sort()
L
# Ex.5
L.sort(reverse=True)
L
2 in L
[1,2,3] in L
L = []
L
L.append(1)
L.append(2)
L.append(3)
L
L = [1,2,3] + [4,5]
L
len(L)
course = {
  "954340": "Enterprise Database",
  "954471": "Business Data Mining",
  "954472": "Business Data Visualization"
}
print(course)
course["954471"]
course["954499"] = "Independent Study"
course
course.keys()
course.values()
list(course.values())
sorted(course)
# EX.6
sorted(list(course.values()))

"954472" in course
"Business Data Mining" in course
course = {}
course.update({"954471":"Business Data Mining"})
course.update({"954472":"Business Data Visualization"})
course
len(course)
userInput = int(input("n = "))
for i in range(0,userInput+1):
    print("."*i)
userInput = input("Sample numbers :  ")
userList = userInput.split(",")
odd = 0
even = 0
for item in userList:
    if int(item)%2 == 0:
        even = even + 1
    else:
        odd = odd + 1
print("Number of even numbers : "+str(even))
print("Number of odd numbers : "+str(odd))
