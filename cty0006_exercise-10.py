# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 01:24:20 2020

@author: cty
"""

#Exercise 10:
str_list = list(input('Input a string: '))


def count(strlist=[]):
 len_lower = 0
 len_upper = 0

 for i in str_list:
    if i.islower():
        len_lower += 1
    elif i.isupper():
        len_upper += 1
        
 return (len_lower,len_upper)

 
a=list(map(int,count(str_list)))
print ('No. of Upper case characters: {}'.format(a[1]))
print ('No. of Lower case characters: {}'.format(a[0]))