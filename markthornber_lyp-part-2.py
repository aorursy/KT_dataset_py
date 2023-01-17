a = 3
b = 4
c = 2
if b**2-4*a*c>0:
    print('Two real roots')
if b**2-4*a*c==0:
    print('Repeated root')
if b**2-4*a*c<0:
    print('No real roots')
print('This is always printed')
a = 3
b = 4
c = 2
disc = b**2-4*a*c
if disc>0:
    print('Two real roots')
elif disc==0:
    print('Repeated root')
else:
    print('No real roots')
print('This is always printed')
score = 63
if score>=78:
    print('A')
elif score>=65:
    print('B')
elif score>=57:
    print('C')
elif score>=49:
    print('D')
elif score>=42:
    print('E')
else:
    print('U')
score = 63
grade = ''
if score>=78:
    grade ='A'
elif score>=65:
    grade ='B'
elif score>=57:
    grade ='C'
elif score>=49:
    grade ='D'
elif score>=42:
    grade ='E'
else:
    grade ='U'
print('Your grade is '+grade)
score = 63
grade = ''
boundaries = {'A':78,'B':65,'C':57,'D':49,'E':42}
if score>=boundaries['A']:
    grade ='A'
elif score>=boundaries['B']:
    grade ='B'
elif score>=boundaries['C']:
    grade ='C'
elif score>=boundaries['D']:
    grade ='D'
elif score>=boundaries['E']:
    grade ='E'
else:
    grade ='U'
print('Your grade is '+grade)
number = 1
while number<11:
    print(number**2)
    number = number+1
number = 1
while number<=10:
    print(number**2)
    number += 1
number = 1
while number < 1000000:
    print(number)
    number *= 2
position = 0
message = 'Hello World'
while position<len(message):
    print(message[position])
    position += 1
list1 = [5,3,27,4]
pos = 0
while pos<len(list1):
    print(list1[pos]**3)
    pos += 1
message = 'Hello World'
for letter in message:
    print(letter)
list1 = [5,3,27,4]
for number in list1:
    print(number**3)
boundaries = {'A':78,'B':65,'C':57,'D':49,'E':42}
for grade in boundaries:
    print(grade)
    print(boundaries[grade])
list2 = [0,1,2,3,4,5,6,7]
for counter in list2:
    print(counter*4)
for i in range(10):
    print(i)
print('-----')
for j in range(3,10):
    print(j)
print('-----')
for k in range(2,30,5):
    print(k)
print('-----')
for l in range(12,0,-2):
    print(l)
squares = [n**2 for n in range(1,51)]
print(squares)
newsequence = [n+3 for n in squares]
print(newsequence)
evensquares = [n for n in squares if n%2==0]
print(evensquares)
password = 'mypassword*2'
uc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lc = 'abcdefghijklmnopqrstuvwxyz'
sp = '*%^$£'
nu = '0123456789'
uctrue = 0
lctrue = 0
sptrue = 0
nutrue = 0
for letter in password:
    if letter in uc:
        uctrue = 1
    if letter in lc:
        lctrue = 1
    if letter in sp:
        sptrue = 1
    if letter in nu:
        nutrue = 1
conditions = uctrue+lctrue+sptrue+nutrue
if len(password)>7 & conditions==4:
    print('Valid')
else:
    print('Invalid')