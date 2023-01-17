# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:47:24 2020

@author: cty
"""

#Exercise 9:
import re

x = float(input('Input a number: '))
numRange = re.findall(r"\-?\d+\.?\d*",input('Input a range (min,max): '))
numRange_float = list(map(float,numRange))
y = max(numRange_float)
z = min(numRange_float)


def if_in_range(a,b,c):
    if ((a>=b) and (a<=c)):
        return print('The number {} is in the range ({},{})'.format(a,b,c))
    else:
        return print('The number {} is not in the range ({},{})'.format(a,b,c))
    
if_in_range(x,z,y)