# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:20:13 2020

@author: cty
"""

#Exercise 6:
a = int(input('Input an intrger: '))
a_root = a**0.5
a_pwr = a**2
if ((1<a_pwr) and (a_pwr<=6) and (a_root ** a_pwr == a)):
    print('The root is {} and the pwr is{}'.format(a_root,a_pwr))
else:
    print('no such pair of root and pwr exists')