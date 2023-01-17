import re

x="I'm born in 25/09/2000 and my favorite number is 45 "

y=re.findall('[0-9]+',x)

print(y)

y=re.findall('[AEIOU]+',x)

print(y)
x='From: USing the : character'   #String



y=re.findall('^F.+:',x)           #GREEDY Matching

print(y)

y=re.findall('^F.+?:',x)          #NON_GREEDY Matching

print(y)
z='From xyz.stanford@abc.ac.za Sat Jan 5 09:15:16 2020'



y=re.findall('\S+@\S+',z)

print(y)

y=re.findall('^From (\S+@\S\S+)',z)

print(y)
atpos=z.find('@')

print(atpos)

spos=z.find(' ',atpos)

print(spos)

host=z[atpos+1:spos]

print(host)
words=z.split()

email=words[1]

same=email.split('@')

print(same[1]) 
y=re.findall('@([^ ]*)',z)

print(y)
x='We just received $10.00 for cookies.'

y=re.findall('\$[0-9.]+',x)

print(y)

import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import re





data=open("/kaggle/input/regular-expression/regular_expression.txt")



sum=0



for line in data:

    line=line.rstrip()

    stuff=re.findall('[0-9]+',line)

    if len(stuff)==0:continue

    for i in stuff:

        sum=sum+int(i)



print("Sum of all the number present in data=",sum)



    
