import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)
a = 'nyc'
b = 'nyc'
print(a)
a = 123
print(a)
print(b)
float_num = 8.3142
print(float_num)
print(type(float_num))
import math as m
u = m.log10(10)
v = m.ceil(9.8)
print(u)
print(v)
(3+2-8+982)/54
l = 5
l += 25
print(l)
a = 23
b = 24
a == 23
ok = True
not ok

type(True)
def fizz_buzz(n=100):
    lst = []
    for i in range(1,n):
        if i % 3== 0 and i % 5 == 0:
            lst.append('Fizz_Buzz')
        elif i % 3 == 0:
            lst.append('Fizz')
        elif i % 5 == 0:
            lst.append('Buzz')
        else:
            lst.append(i)
    print(lst)

fizz_buzz()
    
def area_circle():
    r = float(input('Radius =' ))
    area = (3.142)*(r*r)
    print('Area = ',area)

area_circle()    
import datetime
n = datetime.datetime.now()
print('Current date and time')
print(n.strftime("%d-%m-%y\n%H:%M"))
i =input('First Name : ')
o =input('Last Name: ')
print(o,'',i)
n1 = input('Enter the numbers: ')
list = n1.split(',')
print('List: ',list)
print('Tuple: ',tuple(list))
filename = input('Enter name of the file: ')
list = filename.split('.')
print('Extension of file is: ',list[1])
colour = ['Red','Yellow','Black']
print("%s %s"%(colour[0], colour[-1]))
exam_st_date = (11,12,2014)
print( "The examination will start from : %i / %i / %i"%exam_st_date)
a = int(input("Input an integer : "))
n1 = int( "%i" % a )
n2 = int( "%i%i" % (a,a) )
n3 = int( "%i%i%i" % (a,a,a) )
print (n1+n2+n3)
print(abs.__doc__)
import calendar as c
mnth = int(input('Enter the month:\n'))
yr = int(input('Enter year:\n'))
print('\n',c.month(yr,mnth))
from datetime import date
f_date = date(2014, 7, 2)
l_date = date(2014, 7, 11)
delta = l_date - f_date
print(delta.days)
r = float(input('Radius = '))
vol = (4/3)*3.142*pow(r,3)
print('Volume of sphere with radius %s is %s'%(r,vol))
num =float(input('Enter a number: '))
if num > 17:
    print('Ans = ',(num - 17)*2)
elif num <=  17:
    print('Ans = ',(17 - num))
x = int(input('Enter 1st number = ')) 
y = int(input('Enter 2nd number = '))
z = int(input('Enter 3rd number = '))
if x == y and y == z:
    print('O/P :\n',(x+y+z)*3)
else:
    print('O/P :\n',(x+y+z))
def new_string(str):
    if len(str) >= 2 and str[:2] == "Is":
        return str
    return "Is" + str

print(new_string("Array"))
print(new_string("IsEmpty"))
def larger_string(str, n):
    result = ""
    for i in range(n):
        result = result + str
    return result

print(larger_string('abc', 2))
print(larger_string('.py', 3))
def iseven(num = 0):
    if num % 2 == 0:
        print('%i'%num)
    else:
        print('%i'%num)
iseven(3588)    
lst = [1,2,3,4,5,4,9,4,5,6,32,36,24,4]
count = 0
for i in lst:
    if i == 4:
        count += 1
print(count)
def substr(string ='abc',n = 2):
    if len(string) > 2:
        print(string[:2]*n)
    else:
        print(string*n)
substr('Mayank',4)
def isvowel(s = 'a'):
    ss = s.lower()
    return ss in 'aeiou'
isvowel('A')
def check_num(lst = [1,2,4,5], n = 2):
    return (n in lst)
check_num([2,3,4,6,2210125],-1)
lst = [2, 3, 5, 6]
for i in range(0,len(lst)):
    print(lst[i]*'*')
    
li = ['12','d','fy','hggf']
str1 = ''
for i in li:
    str1 += str(i)
print(str1)
def iseven(num = 0):
    if num % 2 == 0:
        return True
    return False
numbers = [    
    386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345, 
    399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217, 
    815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717, 
    958,743, 527
    ]
# print(237)
for i in numbers:
    if i >= 273 and iseven(i) == True:
        print(i)
        
color_list_1 = set(["White", "Black", "Red"])
color_list_2 = set(["Red", "Green"])
list1=[]
for i in color_list_1:
    if i not in color_list_2:
        list1.append(i)
print(set(list1))
def area_of_triangle(b=1,h=1):
    return (0.5*float(b*h))
area_of_triangle(20,40)
def sum(x=1,y=2,z=3):
    if x == y or y == z or x == z:
        return 0
    return (x+y+z)
print(sum(2, 1, 2))
print(sum(3, 2, 2))
print(sum(2, 2, 2))
print(sum(1, 2, 3))
def test_number5(a=5,b=5):
    if a == b or a-b == 5 or a+b == 5:
        return True
    return False
print(test_number5(7, 2))
print(test_number5(3, 2))
print(test_number5(2, 2))
def check_type(u=0,v=1):
    '''
It takes two arguments 
It is used to check the type and *ADD*
    '''
    if isinstance(u,int) and isinstance(v,int):
        return u+v
    raise TypeError('Input should be integers')
check_type(1,56894)
help(isinstance)
def personal_details():
    name, age = "Mayank", 20
    address = "Mumbai, Maharashtra, India"
    print("Name: {}\nAge: {}\nAddress: {}".format(name, age, address))

personal_details()

