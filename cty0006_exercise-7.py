# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:53:44 2020

@author: cty
"""

#Exercise 7:   
x = float(input('Input a number: '))

def cube_root(a):
 epsilon = 0.01
 numGuesses = 0
 low = 0.0
 high = max (1.0, a)
 ans = (high + low)/2.0
 while abs(ans**3 - a) >= epsilon:
      print ('low =', low, 'high = ', high, 'ans =', ans)
      numGuesses += 1 
      if ans**3 < a:
           low = ans
      else:
           high = ans
      ans = (high + low)/2.0
      if abs(ans**3 - a) < epsilon:
          print ('low =', low, 'high = ', high, 'ans =', ans)
          numGuesses += 1 
          print ('numGuesses =', numGuesses)
          return ans
 
if x < 0:
   x = -x
   print(-cube_root(x), 'is close to cube root of',-x)
else:
   print(cube_root(x), 'is close to cube root of',x)