print("Hello Come to Delhi")
print("another fun message")
# the below code is used to print 'welcome Jyothi R' < - description

print("welcome Jyothi R")
# use shift+enter as  short cut to execute cellb

print('I love SBT Forever')
print("shortcut-> shift+enter to execute current cell film")
#Variables declaration

name="Sailesh Praveen M T"
print(name)
age=21
print(name , ' is ',age) # constructing information out of variables..
# Type Identification
taxRate=9.5
print(  type(taxRate)  )  # type function returns the type of variable.
# Understanding Operators
# Arithmatic Operator



# + , /  , - , * , %
print('5+2',5+2)

print('5-2',5-2)

print('5/2',5/2)

print('5*2',5*2)

print('5%2',5%2) # %remainder
# Comparision Operator



# > , < , >= , <= ,!= , ==
print(50<=2)
print(5==4)
print(5==5)
# member ship operator in ,not in
bucket=['apple','mango','cherry']
'mango' in bucket
'pineapple' in bucket
# Loop in Python

for  n in range(3):

    print("hi")



print("this won't be repeated")

#Indent->Scope
# While Loop



meter=1 



while(meter<=10):

    print(meter)

    meter=meter+2
# Declaring a Function?



def greet():

    print("Greetings from din")

#call the function

greet()
greet()
for n in range(5):

    greet()
def getSquare(n): #Parameter

    print(n*n)
getSquare(5)
getSquare(6)
def calc(n):

    print('square fancy symbol ',n*n)

    print('cube - >',n*n*n)
calc(3)
calc(9)
for x in range(2,10):

    if x==5:

        break

    print(x)