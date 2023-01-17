# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:01:32 2020

@author: cty
"""

#Exercise 5:
a = int(input('Input the 1st number: '))
b = int(input('Input the 2nd number: '))
c = int(input('Input the 3rd number: '))
d = int(input('Input the 4th number: '))
e = int(input('Input the 5th number: '))
f = int(input('Input the 6th number: '))
g = int(input('Input the 7th number: '))
h = int(input('Input the 8th number: '))
i = int(input('Input the 9th number: '))
j = int(input('Input the 10th number: '))
numList = []
numList.append(a)
numList.append(b)
numList.append(c)
numList.append(d)
numList.append(e)
numList.append(f)
numList.append(g)
numList.append(h)
numList.append(i)
numList.append(j)
even_numList = []
for x in numList:
   if x%2 == 0:
     even_numList.append(x)
if even_numList:
    Max = max(even_numList)
    print('The biggest even number is {}.'.format(Max))
else:
    print('There is no biggest even number')