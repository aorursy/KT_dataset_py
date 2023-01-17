xyz = '100'
print(xyz)
xyz = '250'
print(xyz)
a = 10 
a = a - 1
print(a)
A = 10 
print(A)
A = A + 100
print(A)
A = 3 * A + A
print(A)
x = 4 

while x > 0 :
    print(x)
    x = x-1 
print('Done')

print('=====')

print('Now The value of x is ',x)
n = 4

while n > 0:
    print('Raja')
    print('Ram')
print('done')
n = 6
while n > 0:
    n -= 1
    if n == 2:
        break
    print(n)
print('Loop breaked')
n = 6
while n > 0:
    n -= 1
    if n == 2:
        print('Here the loop will infinate and Loop Breaks at 2')
        continue
    print(n)
print('Loop ended.')
n = 6
while n > 0:
    n -= 1
    print(n)
else:
   print('Loop ended')
for varname in sequence:
    codeblock 
for i in [1,2,3,4,5]:
    print(i)
for letter in 'Hello':
    print(letter)

names = ['Raja','Ram','Sita']

for i in names:
    print('Happy Birthday',i)
for i in range(5):
    print(i)
for i in range(3,7):
    print(i)
counter = 0 

for i in [10,9,8,6,2,1,0]:
    counter = counter + 1
    print(counter,i)

total = 0 

for i in [10,9,8,6,2,1,0]:
    total = total + i
    print(total,i)
Maximum & Minimum
Maximum = None

print('When the program executes, the output is as follows:')
for i in [100, 110, 99, 769, 960, 12]:
    if Maximum is None or i > Maximum :
        Maximum = i
    print(i,Maximum)
        
print('Maximum number in the list is', Maximum)
Minimum = None

print('When the program executes, the output is as follows:')
for i in [100, 110, 99, 769, 960, 12]:
    if Minimum is None or i < Minimum :
        Minimum = i
    print(i,Minimum)
        
print('Minimum number in the list is', Minimum)
def min(values):
    Minimum = None

    for i in values:
        if Minimum is None or i < Minimum :
            Minimum = i
    print('Minimum number in the list is', Minimum)
min([100, 110, 99, 769, 960, 12])
XYZ = "Python"

First_letter = XYZ[1]

print(First_letter)
XYZ = "Python"

First_letter = XYZ[0]

print(First_letter)
XYZ = "Python"

XYZ[1.0]
XYZ = "Python"

Last_letter = XYZ[-1]
Second_last_letter = XYZ[-2]

print(Last_letter,Second_last_letter)
course_name = 'Python'

len(course_name)
course_name = 'Python'

print(course_name[len(course_name)])
fruit = 'APPLE'
index = 0

while index < len(fruit):
    letter = fruit[index]
    print(letter)
    index = index + 1
name = "RajaRam"
index = 0

for eachchar in name:
    if index < len(name):
        eachchar = name[index]
        print(eachchar)
        index = index + 1
name = 'RajaRam'
name[0:3]
name = 'RajaRam'
name[0:4]
name = 'RajaRam'
name[0:4]
name = 'RajaRam'
name[:4]
name = 'RajaRam'
name[4:]
name = 'RajaRam'
name[4:3]
name = 'RajaRam'
name[4:4]
name = 'RajaRam'
name[0] = 'J'
name = 'RajaRam'
name = 'J' + name[1:]
print(name)
a = 'Hello '
b = 'This is python tutorial '
c = 'for learning iteration and strings concept'

Greeting = a + b + c
print(Greeting)
name = 'RajaRam'

counter = 0
for i in name :
    if i == 'a':
        counter = counter + 1
print(counter)
name = 'RajaRam'
'a' in name
name = 'RajaRam'
'b' not in name
a = "Raja"
b = "Raja"
c = "raja"

a == c
a = "Ra"
b = "raja"

a < b
name = "RajaRam"
dir(name)
name = 'ramaraju is good boy'
name.capitalize() 
name = "Raghu"
name.islower()
name = "RAGHU"
name.isupper()
name = 'ramaraju is good boy'
name.endswith('boy')
name = "RAGHU"
name.lower()
name = "Raghu"
name.upper()
name = 'ramaraju is good boy'
name.title()
name = 'Ramaraju Is Good Boy'
name.swapcase()
name = 'Ramaraju_is_a_good_boy'
name.split('_')
name = 'Ramaraju is a good boy'
name.split(' ')
email = "ramaraju@gmail.com"
email.find('@')
Str1 = print(email[9:])
Str2 = print(email[:8])
email = "ramaraju@gmail.com"
email.split('@')
name = 'Python is an interpreted, high-level and general-purpose programming language'
name.find(' ')
data = ('Python' ,1980)
print('%s was created in the late %d s as a successor to the ABC language' % data)
print('%s s age is %d' % ('Raja',18))
print('%d s age is %d' % ('Raja',18))