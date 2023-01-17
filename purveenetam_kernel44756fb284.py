j = input('Enter a number:')
i = 1
while i<=10:
    print(j+' * %d ='%i,j*i)
    i+=1
print('Enter an alpahbet, dont use special characters like # , $ % ^ & * ')
alpha = input('Input any alphabet : ')
if alpha == ('a' or 'e' or 'i' or 'u' or 'A' or 'E' or 'I' or 'O' or 'U'):
    print('%s is a vowel.' %alpha)
else:
    print('%s is a consonant.' %alpha)
print('Enter an alpahbet, dont use special characters like # , $ % ^ & * ')
alpha = input('Input any alphabet : ')
if alpha == ('a' or 'e' or 'i' or 'u' or 'A' or 'E' or 'I' or 'O' or 'U'):
    print('%s is a vowel.' %alpha)
else:
    print('%s is a consonant.' %alpha)
x = input('Enter the length of first side of a triangle: ')
y = input('Enter the length of second side of a triangle: ')
z = input('Enter the length of third side of a triangle: ')
if x==y==y==z==z==x:
    print('This is an equilateral triangle')
elif  x!=y==y!=z==z!=x:  
    print('This is a scalene triangle')
elif x<=y or x>=y or y<=z or Y>=z or z<=x or z>=x or x==y or y==z or z==x: 
    print('This is an isoscles triangle')
x = input('Enter the length of first side of a triangle: ')
y = input('Enter the length of second side of a triangle: ')
z = input('Enter the length of third side of a triangle: ')
if x==y==y==z==z==x:
    print('This is an equilateral triangle')
elif  x!=y==y!=z==z!=x:  
    print('This is a scalene triangle')
elif x<=y or x>=y or y<=z or Y>=z or z<=x or z>=x or x==y or y==z or z==x: 
    print('This is an isoscles triangle')
x = input('Enter the length of first side of a triangle: ')
y = input('Enter the length of second side of a triangle: ')
z = input('Enter the length of third side of a triangle: ')
if x==y==y==z==z==x:
    print('This is an equilateral triangle')
elif  x!=y==y!=z==z!=x:  
    print('This is a scalene triangle')
elif x<=y or x>=y or y<=z or Y>=z or z<=x or z>=x or x==y or y==z or z==x: 
    print('This is an isoscles triangle')
p = int(input('Enter perpendicular : '))
b = int(input('Enter Base : '))
h = (p**2 + b**2)**0.5
print("hypotenuse : ", int(h))