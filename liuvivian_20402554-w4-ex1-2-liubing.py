'''

import decimal

from decimal import(Decimal)

decimal.getcontext().prec=9

'''

x=input('pls input your number:')

x=float(x)

km=x/0.62137

meters=1000*km

'''method 1

km=Decimal(x)/Decimal(0.62137)

meters=1000*km

print (meters)

print (km)

print(x,end=' ')

print('miles is equivalent to',end=' ')

print(km,end=' ')

print('km/',end=' ')

print(meters,end=' ')

print('meters')

'''

#method 2 use {:.f}.format function

print('{:.2f}miles is equvalent to'.format(x),end=' ')

print('{:.4f}km/{:.1f}meters'.format(km,meters))
name=input("what's your name:")

age = input("What's your age:")

age=int(age)+2047-2020

print('Hi',str(name),'!In 2047 you will be '+str(age)+'!')