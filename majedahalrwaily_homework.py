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
#Execute the following instructions: 
a=12
b= 34
c= "Python"
d= "Programming"
print(c+d, a+b) 
print(a+b, a-b, a/b, a//b, a**b)

#Give output of the following instructions:
print(not(a==b)) 
print((a==b)& (not(a>b)))
print((a==b) or(b<a))
#Write instruction to compute the number of occurrence of letter p in the string “python programmingp ”

string = "python programmingp "
substring = "p"
count = string.count(substring)
print( "The count is:" , count) 

#Write instruction to replace letters p in in the string “python programmingp” by the letter “P”

s = "python programmingp"
print(s.replace( "p", "P" )) 

#Create the following list1: ['st1', 'st2', 'st3', 'st4', 'st5', 'st6']

list1= ["st1", "st2", "st3", "st4", "st5", "st6"]
print (list1)
#Create a string S in which you concatenate the fourth and fifth elements of the previous list
S = [list1[3], list1[4]]  
print (S)
#Create the following list: list2 ['id1', 'id2', 'id3', 'id4', 'id5', 'id6']
list2 = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6']
print (list2)
#Create list 3 which contains list 1 and list2
list3 = [list1, list2 ]
print (list3)
#Create the following string: course="This is Data Science Programming Course"
course="This is Data Science Programming Course"
print (course)
#Create list4 by splitting the string course 
list4 = []
list4 = course.split()
print (list4)
#Add the following list in the end of list4 ['this','is','python','primer','chapter']
list4.extend(['this','is','python','primer','chapter'])
print(list4)
#Count the occurrence of string ‘is’ in list4
substring = "is"
count = list4.count(substring)
print( "The count is:" , count) 
#Reverse elements of list4
list4.reverse()
print (list4)
#Sort and print list4
list4.sort()
print (list4)

#sort the list in increasing order of string lengths
list4.sort(key = len) 
print(list4)