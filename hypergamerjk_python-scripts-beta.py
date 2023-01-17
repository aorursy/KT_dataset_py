if 4+2==5:
    print('true')
elif 4+2==6:

    print(4+2)
x1 = 3
x2 = 3

if x1>x2:
    print('%g is greater than %g' %(x1,x2))
elif x1<x2:
    print('%g is greater than %g' %(x2,x1))
else:
    print('%g is equal to %g' %(x2,x1))
from IPython.display import display,Math

for i in range(0,4):
    for j in range(0,5):
        if i>0 and j>0:
            display(Math('%g^{-%g} = %g' %(i,j,i**-j)))
a = -4
b = abs(a)
a,b
from IPython.display import display,Math

x = 9
display(Math('|%g| = %g' %(x,abs(x))))
numbers = [-4,-6,-1,43,-18,2,0]

# for-loop over the numbers
for numi in numbers:
    if numi<-5 or numi>2:
        print('Absolute value of %g is %g.' %(numi,abs(numi)))
    else:
        print( str(numi) + ' was not tested.')
        
a = 10
b = 3

# division
int(a/b)
a = 38914
b = 316734

divis = int(a/b)
remainder = a%b

print('%g goes into %g, %g times with a remainder of %g' %(b,a,divis,remainder))
nums = range(-5,6)

for i in nums:
    
    firstchar = ' '
    if i<0:
        firstchar = ''
    
    # test and report
    if i%2 == 0:
        print('%s%g is an even number' %(firstchar,i))
    else:
        print('%s%g is an odd  number' %(firstchar,i))
def computeremainder(x,y):
    divis = int( x/y )
    remainder = x%y
    
    print('%g goes into %g, %g times with a remainder of %g' %(y,x,divis,remainder))
computeremainder(100,6)
def divisionWithInput():
    
    x = int( input('Input the numerator: ') )
    y = int( input('Input the denominator: ') )
    
    divis = int( x/y )
    remainder = x%y
    
    print('%g goes into %g, %g times with a remainder of %g' %(y,x,divis,remainder))
divisionWithInput()
# create the functions

def powers(x,y):
        display(Math('x^{y} = z'))