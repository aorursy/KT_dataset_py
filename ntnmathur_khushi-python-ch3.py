def calcSomething(x):
    r=2*x**2
    return r
a=int(input("Enter a number:"))
print(calcSomething(a))
# <function-name>(<value passed as argument>)

"""
1. Write a function that takes amount in dollars and dollar in ruppee conversion price, it then returns amount converted to ruppees.
Create in both void and non-void forms
""" 

# void
def dollarToRuppeesVoid(dollars, conversion):
    ruppees=dollars*conversion
    print(ruppees)
    
# Non-void
def dollarToRuppees(dollars, conversion):
    ruppees=dollars*conversion
    return ruppees


# void
# d=int(input("Enter the price in dollars:"))
# c=int(input("Enter the conversion rate:"))
# dollarToRuppeesVoid(d,c)

# Non-void
d=int(input("Enter the price in dollars:"))
c=int(input("Enter the conversion rate:"))
print(dollarToRuppees(d,c))
    
"""
2.Write function to calculate volume of a box with appropriate default values for its parameters. Your function should have the following input parameters:
- length of box
- width of box
- height of box
Test it by writing complete program to invoke it.
"""
def volume(length=5, width=2, height=7):
    volume=length*width*height
    return volume

l=int(input("Enter length of box:"))
w=int(input("Enter width of box:"))
h=int(input("Enter height of box:"))
vol=volume(l,w,h)
print("Volume:",vol)
"""
3. Write a program to have the following functions:
(i) a function that takes a number as the argument and calculates the cube for it. The function does not return a value. If there is no value passed to the function,
the function should calculate cube of 2.
(ii) a function that takes two char arguments and returns True if both the arguments are equal otherwise false.
Test both these functions by giving appropriate function call statements.
"""
def cube(num=2):
    cube=num**3
    print("Cube of number:",cube)

    
n=int(input("Enter a number:"))
cube(n)

def compare(char1, char2):
    if char1==char2:
        return True
    else:
        return False
    
c1=input("Enter character 1:")
c2=input("Enter character 2:")
result=compare(c1,c2)
print(result)
import random
random.randint(3,7)
"""
4.
"""
import random
def findRandom(num1, num2):
    r1=random.randint(num1,num2)
    r2=random.randint(num1,num2)
    r3=random.randint(num1,num2)
    return r1,r2,r3

num1=int(input("Enter number 1:"))
num2=int(input("Enter number 2:"))
r1,r2,r3=findRandom(num1,num2)
print("The random numbers between {num1} and {num2} are {r1},{r2} and {r3}".format(num1=num1,num2=num2,r1=r1,r2=r2,r3=r3))
"""
5.
"""
def compare(s1,s2):
    if len(s1)==len(s2):
        return True
    else:
        return False

s1=input("Enter string 1: ")
s2=input("Enter string 2: ")
result=compare(s1,s2)
print(result)
"""
6.
"""
def nthRoot(x,n=2):
    r=x**(1/n)
    return r

x=int(input("Enter the first number:"))
n=int(input("Enter the second number:"))

root=nthRoot(x,n)
print("{n}th root of {x} is {root}".format(n=n,x=x,root=root))
num=3
u=(10**num)-1
l=10**(num-1)
print(u,l)
"""
7.
"""
# num=3
# u=(10**num)-1
# l=10**(num-1)
# print(u,l)


import random
def findRandom(n):
    upper=(10**n)-1
    lower=10**(n-1)
    r=random.randint(lower,upper)
    return r

num=int(input("Enter the number:"))
rm=findRandom(num)
print("The result is {res}".format(res=rm))
"""
8.
"""

def findMinOnes(n1,n2):
    rem1=n1%10
    rem2=n2%10
    if rem1<rem2:
        return n1
    else:
        return n2
    
num1=int(input("Enter number 1:"))
num2=int(input("Enter number 2:"))
print("Number with minimum ones value is {num}".format(num=findMinOnes(num1,num2)))
"""
9.
"""
def find4nums(a,b):
    if a>b:
        low=b
        high=a
    else:
        low=a
        high=b
    print("low:",low)
    print("high:",high)
    diff=(high-low)/3
    print("diff:",diff)
    i=low
    lst=[]
    while i<=high:
        print(i)
        lst.append(i)
        i=i+diff
    return lst

n1=float(input("Enter first num:"))
n2=float(input("Enter second num:"))
lst=find4nums(n1,n2)
print(lst)
import random
15 + random.random()*(35-15)
import urllib
urllib.request.urlopen("http://python.org/").read()
