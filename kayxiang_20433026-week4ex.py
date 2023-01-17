# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 

@author: XIANG Kai
"""
#Finger Exercise 1
mile = float(input("Please input the miles:"))       #ask user to input the number of miles
mile_km = mile/0.62137                               #mile converted to km
mile_meter = 1000*mile_km                            #mile converted to meter
print('{:.2f} miles is equivalent to '.format(mile))
print('{:6.4f}km / {:6.1f}meters'.format(mile_km , mile_meter))

#Finger Exercise 2
name = input('What is your name?')             #ask the user to input the name 
age_now = int(input('What is your age?'))      #ask the user to input age for now
age_2047 = age_now + 27                        #caculate age in 2047
print('Hi {}! In 2047 you will be {:d}!'.format(name ,age_2047 ))   






#Finger Exercise 3
x = int(input('please input x:'))
y = int(input('please input y:'))
z = int(input('please input z:'))

if x%2 == 0:                #if x is even
    if y%2 == 0:            #if y is even on the condition of x is even
        if z%2 == 0:        #if z is even on the condition of x and y are even
            print('None of them are odd number')
        else:               #if z is odd on the condition of x is even and y is even
            print('The largest odd number is ',z)
    else:                   #if y is odd on the condition of x is even
        if z%2 == 0:      
            print('The largest odd number is ',y)
        else:
            largest_num = max(y,z)
            print('The largest odd number is ',largest_num)
else:
    if y%2 == 0:
        if z%2 == 0:
            print('The largest odd number is ',x)
        else:
            largest_num = max(x,z)
            print('The largest odd number is ',largest_num)
    else:
        if z%2 == 0:
            largest_num = max(x,y)
            print('The largest odd number is ',largest_num)
        else:
            largest_num = max(x,y,z)
            print('The largest odd number is ',largest_num)
 
#Finger Exercise 4
numXs = int(input("How many times should I print the letter x?"))
toPrint = ""
num = 0
while num<numXs:
    toPrint+="X"
    num+=1
print(toPrint)


#more simple method
# numXs = int(input("How many times should I print the letter x?"))
# toPrint = numXs * "X"
# print(toPrint)