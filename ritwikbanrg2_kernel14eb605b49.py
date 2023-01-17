# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(1)
print(2)
print(3)
a=int(input("Enter number"))
a*2
print("sd",6)
a=5
print("Number: ",a)
print("Top \n Bottom")
a=input("Enter Name")
b=input("Enter age")
c=input("Enter GPA")
d=input("Enter branch")
s="Name: "+a+"\nAge:"+b+" GPA:"+c+"\nBranch:"+d
print(s)
l=[]
l.append(s)
print(l[0])
a=[2,3,4]
print(a)
a[1]
arr=[1,3,5.8,"ABC"]
type(arr[3])
arr2=[1,2,5,[3.4,5.6,"To"],10]
arr2[3][1]
arr
arr.append(3)
arr
arr[3:5]
a={"One":1,"Three":3,"Five point Eight":5.8}
a["Three"]
type(a.values())
x=list(a.keys())
y=a.keys()
type(x)
type(y)
y[1]
x[1]
l1=["Number","Square","Cube"]
l2=[3,9,27]
d=dict(zip(l1,l2))
print(d)
print(s)
l=[]
l.append(s)
print(l[0])
a=7/2
b=7-2
c=7+2
d=7*2
print(a,b,c,d)
x=9/2
y=9//2
print(x,"   ",y)
z=13%5
print(z)
52783278403%751
a=int(input("Enter number 1"))
b=int(input("Enter number 2"))
print(a%b)
3*7
7**3
24!=(5*6)
z+=4
print(z)
#z+=4                  z=z+4         z*=6
z*=6
print(z)
z/=2
print(z)
5&7
(5==(2+3))and(7>9)
a=7
print(-a)
5<<2
5*4
60>>2
y=int(input("Enter year"))
if(y%400==0):
    print("Leap year")
elif(y%100==0):
    print("Not leap year")
elif(y%4==0):
    print("Leap year")
else:
    print("Not Leap Year")
n=int(input("number of rows"))
for i in range (0,n):
    for j in range (0,i+1):
        print("*",end=' ')
    print(" ")
for i in range (1,9):
    for j in range (1,5):
        print(j,end=' ')
    print("")
a=[2,3,4,22,33,21,55,72]
for i in a:
    if (i%2==0):
        print("Even")
    else:
        print("Odd")
a=[]
while(True):
    x=int(input("Enter next value"))
    if(x<=0):
        break
    a.append(x)
a
a1=[]
for i in range(0,5):
    x=int(input("Enter a number: "))
    a1.append(x)
print(a1)
arr=[]
c=1
x=int(input("Enter the first element: "))
arr.append(x)

while (c<5):
    x=int(input("Enter next element: "))
    if(x<=arr[c-1]):
        print("You must enter values in sorted order")
        continue
    arr.append(x)
    c+=1
print(arr)
n=0
d=3
rev=47823
n=32874
s=0
d=0
while(n>0):
    d=n%10
    s+=d
    n=n//10
print(s)
l=[2,3,4,2,3,2,4,5,3]
p=1
for i in l:
    p*=i
print(p)
n=4389
rev=0
d=0
while(n>0):
    d=n%10
    rev=(rev*10)+d
    n=n//10
print(rev)
n=int(input("Enter number"))
p=1
for i in range(1,n+1):
    p*=i
print(p)
l1=[1,2,3,1,45,6]
l2=[23,4,1]
l1.count(1)
l1.extend(l2)
print(l1)
l=[[3,'ade'],[22,'ffr'],[10,'poe'],[72,'abo']]
print(sorted(l,key=givestring))
def givelen(l1):
    return(len(l1))
l=['fdsugf','fdsugfi','fd','qweo','eiowquosreewui','qaoo']
print(sorted(l,key=givelen))
len("fjdijfaeo")
l=[2,3,4,5,6]
t=(2,3,4,5,6)
l[3]=9
print(l)
t[3]=9
print(t)
x=int(input("number   "))
s=d=0
t=x
while(t>9):
    s=0
    while(t>0):
        d=t%10
        s=s+d
        t=t//10
    t=s
if(t==1):
    print("Magic")
else:
    print("Not magic")
