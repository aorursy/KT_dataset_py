# Consider two inputs x and y
x = 10
y = 20
print('value of x is', x)
print('value of y is', y)
# x + y = sum of x and y
print('Sum of x and y is', x+y)
# x - y = difference of x and y
print('Difference of x and y is',x-y)
# x * y = product of x and y
print('Product of x and y is',x*y)
# x / y = quotient of x and y
print('Quotient of x and y is',x/y)
#x // y = floored quotient of x and y
print('Floored quotient is',x//y)
#x % y = remainder of x / y
print('Remainder of x/y is',x/y)
# -x = x negated
print('Negated value of x is',-x)
# +x = x unchanged

# abs(x) = absolute value or magnitude of x

# int(x) = x converted to integer
print('Converted to int',int(10.89))
# float(x) = x converted to floating point
print('Converted to floating point',float(x))
# complex(re, im) = a complex number with real part re, imaginary part im. im defaults to zero.

# c.conjugate() = conjugate of the complex number c

# divmod(x, y) = the pair (x // y, x % y)

# pow(x, y) = x to the power y
print('x to the power y',pow(x,y))
# x ** y = x to the power y




# x or y = if x is false, then y, else x
x or y
# x and y = if x is false, then x, else y
x and y
# not x = if x is false, then True, else False
not x

# "<" is strictly less than

# "<=" is less than or equal

# ">" is strictly greater than

# ">=" is greater than or equal

# "==" is equal

# "!=" is not equal

# "is" is object identity
# An is expression evaluates to True if two variables point to the same (identical) object.
# An == expression evaluates to True if the objects referred to by the variables are equal (have the same contents).

# "is not" is negated object identity


# Finding the type of a variable
print('Type of a is',type(x))
type(x)
# Auto typecasting
# Order for auto typecasting : Bool -> int -> float -> str
print(x+0.00)
#Forced typecasting
print(int(1.1))
# True = Anything not equal to zero
# False = 0
print(False + x)
print(True + x)
print(bool(-18))
print(type(int('9')))
# int('8.9') # this throws an error
# Guess the answer for below arithmetic equation
print(5.5 + bool(-1) + int('3')+int(5.6)+ bool(0))

a = 'Data science'
b = 'Machine Learning'
print(a)
print(b)
# [] --> selector operator.
# Slicing Operator : Gives you range
a[0]
a[:]
a[0:]
a[1:]
a[:2]
a[:-1]
a[-2:]
a + ' and ' + b
a[::1]
a[0:10:2]
# 2 --> skips the alternate characters
str(b'Zoot!')
len(a)
# Importing datetime module from python
import datetime
print(datetime.date.today())
#The largest year number allowed in a date or datetime object. MAXYEAR is 9999.
print(datetime.date.max)
#The smallest year number allowed in a date or datetime object. MINYEAR is 1.
print(datetime.date.min)
# Printing the current date and time.
currentDateTime = datetime.datetime.now()
print(currentDateTime)
currentDateTime.date()
currentDateTime.time()
# To get year from a date time object
year  = currentDateTime.year
print('The year of the datetime object is',year,'of data type',type(year))
# To get month from a date time object
month  = currentDateTime.month
print('The month of the datetime object is',month,'of data type',type(month))
# To get day from a date time object
day  = currentDateTime.day
print('The day of the datetime object is',day,'of data type',type(day))
# To get hour from a date time object
hour  = currentDateTime.hour
print('The hour of the datetime object is',hour,'of data type',type(hour))
# To get minutes from a date time object
minutes  = currentDateTime.minute
print('The minutes of the datetime object is',minutes,'of data type',type(minutes))
# To get seconds from a date time object
seconds  = currentDateTime.second
print('The seconds of the datetime object is',seconds,'of data type',type(seconds))
# To get micro seconds from a date time object
microSeconds  = currentDateTime.microsecond
print('The micro seconds of the datetime object is',microSeconds,'of data type',type(microSeconds))
# A duration expressing the difference between two date, time, or datetime instances to microsecond resolution is timedelta.
Days100FromNow = currentDateTime + datetime.timedelta(days=100)
print(Days100FromNow)
# You cant use timedelta on time alone you should have date for that.
newTime = currentDateTime + datetime.timedelta(days=100,hours = 2,minutes=40,seconds = 100)
newTime
# We can use this method for this purpose
def add_to_time(time_object,time_delta):
    import datetime
    temp_datetime_object = datetime.datetime(500,1,1,time_object.hour,time_object.minute,time_object.second)
    #print(temp_datetime_object)
    return (temp_datetime_object+time_delta).time()
from datetime import date
print(date.today())
print(date.weekday(date.today()))

#Return the day of the week as an integer, where Monday is 1 and Sunday is 7
print(date.isoweekday(date.today()))
# declaring a date
d1 = datetime.date(2016,11,24)
d2 = datetime.date(2017,10,24)
max(d1,d2)
print(d2 - d1)
century_start = datetime.datetime(2001,1,1,0,0,0)
time_now = datetime.datetime.now()
time_since_century_start = time_now - century_start
print("days since century start",time_since_century_start.days)
print("seconds since century start",time_since_century_start.total_seconds())
print("minutes since century start",time_since_century_start.total_seconds()/60)
print("hours since century start",time_since_century_start.total_seconds()/60/60)
# conversion of date to string
now = datetime.datetime.now()
string_now = datetime.datetime.strftime(now,'%m/%d/%y %H:%M:%S')
print(now,string_now)
print(str(now))
#Conversion of string to date
date='01-Apr-03'
date_object=datetime.datetime.strptime(date,'%d-%b-%y')
print(date_object)
# Anything other than zero in a if statement is true.
if 0:
    print('ZERO is always false')
if not 0:
    print('Anything other than zero is true')
if True:
    print('')
elif True:
    print()
else:
    print()
def method1():
    10/20
print(method1())
# A function can have function arguments.
def arithmeticOperations(x,y=10):
    return x+y,x-y,x*y,x/y
add, dif, mul, div = arithmeticOperations(10,20)
print(add)
print(dif)
print(mul)
print(div)

result = arithmeticOperations(10,20)
print(result)
# A function can have function arguments.
def FunctionalDef(x,y,function):
    return function(x,y)
FunctionalDef(10,20,arithmeticOperations)
# Empty list
list1 = list()
list1
# Empty list
list2 = []
list2
list1.append(1)
list1
list3 = [1,2,3,4]
list3
# add the lements of other list
list4 = [5,6,7,8]
list4.extend(list3)
list4
element = list4.pop(4)
element
list4
list4.remove(5)
list4
# Iterating a list by elements in the list
for element in list4:
    print (element)
# Iterating a list by index of the list
for index in range(len(list4)):
     print (list4[index])
# A list is mutable
list4[5] = 8
list4.append('a')
list4
x=range(10)
x
# Empty tuple
tuple1 = tuple()
tuple1
# Tuples are immutable
tuple1=(1,2)
tuple1
tuple1[1]
#tuple1[1] = 9
# Empty set
set1 = set()
set1
# You can only pass 1 argument in add method.
set1.add(1)
set1
set1.clear()
set1
set3 = {1,2,3,4,5}
set3
set3.add(2)
set3
set3.discard(2)
set3
dict1 = dict()
dict1
dict1['name'] = 'John'
dict1
age = dict1.get('age')
type(age)
# This throws an error
#dict1['age']



