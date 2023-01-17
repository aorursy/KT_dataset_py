# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:02:47 2020

@author: cty
"""

#Exercise 3:
a = int(input('Input the 1st number: '))
b = int(input('Input the 2nd number: '))
c = int(input('Input the 3rd number: '))
if ((a%2==0) and (b%2==0) and (c%2==0)):
    print('None of them is an odd number.')
else:
    numList = []
    numList.append(a)
    numList.append(b)
    numList.append(c)
    odd_numList = []
    for i in numList:
     if i%2 != 0:
      odd_numList.append(i)
    Max = max(odd_numList)
    print('The biggest odd number is {}.'.format(Max))
    

#Exercise 4
numXs = int(input('How many times should I print the letter X?'))
toPrint = str()
i = 0
while i<numXs:
    toPrint = toPrint + 'X'
    i = i+1
print(toPrint)