def quadSolve(a,b,c):
    disc = b*b-4*a*c
    root1 = (-1*b-disc**0.5)/(2*a)
    root2 = (-1*b+disc**0.5)/(2*a)
    return root1,root2
a = 2
b = 7
c = 3
print(quadSolve(a,b,c))
answer = quadSolve(2,7,3)
print(answer[0])
print(answer[1])
x1,x2 = quadSolve(2,7,3)
print(x1)
print(x2)
print(quadSolve(2,7,3))
n1 = 3
n2 = 9
brian = 5
print(quadSolve(n1,n2,brian))

print(quadSolve(2,1,5))
root1,root2 = quadSolve(2,7,3)
print(root1,type(root1))
root3,root4 = quadSolve(2,1,5)
print(root3,type(root3))
print(root3.real)
print(root3.imag)
def greet(name):
    print('Hello '+name)
    print('How are you?')

greet('Bob')
greet('Sue')
greet('Quadratic Formula')
print('Enter your name')
name1 = input()
greet(name1)
print('Try again')
name2 = input()
greet(name2)
def validPassword(passwordtry):
    uc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lc = 'abcdefghijklmnopqrstuvwxyz'
    sp = '*%^$£'
    nu = '0123456789'
    uctrue = 0
    lctrue = 0
    sptrue = 0
    nutrue = 0
    for letter in passwordtry:
        if letter in uc:
            uctrue = 1
        if letter in lc:
            lctrue = 1
        if letter in sp:
            sptrue = 1
        if letter in nu:
            nutrue = 1
    conditions = uctrue+lctrue+sptrue+nutrue
    if len(passwordtry)>7 & conditions>=3:
        return True
    else:
        return False
    
password = 'bad'
while not validPassword(password):
    print('Passwords must be at least 8 characters')
    print('They must contain at least 3 of: upper case, lower case, numbers and special characters')
    print('Enter your password')
    password = input()

print('Password OK')
    
def square(x):
    return x*x

def makeTable(func,start,end,step):
    for i in range(start,end+1,step):
        print(i,func(i))

makeTable(square,1,10,1)
def makeTable2(func,start,end,step=1):
    print('  x  | f(x) ')
    print('___________')
    for i in range(start,end+1,step):
        print(str(i).ljust(5)+'|'+str(func(i)).rjust(5))
        
makeTable2(square,1,10)

def cube(x):
    return x*x*x

makeTable2(cube,2,20,2)
def fact(n):
    if n>1:
        return n*fact(n-1)
    else:
        return 1

makeTable2(fact,1,5)
def perm(string):
    n = len(string)
    if n == 1:
        return [string]
    else:
        return [string[i]+j for i in range(n) for j in perm(string[0:i]+string[i+1:n+1])]

print(perm("worlds"))