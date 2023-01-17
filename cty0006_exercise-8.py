# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:57:04 2020

@author: cty
"""

#Exercise 8:
import re

def list_multiply(num_list=[]):
    product = 1
    for i in num_list:
        product = product * i
    return product


aList_str = re.findall(r"\-?\d+\.?\d*",input('input a number list: '))
aList_float = list(map(float,aList_str))
print('The result is: {}'.format(list_multiply(aList_float)))