# Tuple

t1 = 12, 23, 34, 45, 21, 89

print (t1)
t2 = (12, 23, 34, 45, 21, 89)

print(t2)
# List

l1 = [12, 23, 34, 45, 21, 89]

print(l1)
l1[:]
len(l1)
type(l1)
# Applying conditions on list/tuple

l1 = [12, 13, 134, 43, 65, 76, 34, 87, 98, 65, 42, 94, 50]
for x in range(0, len(l1)):

    if l1[x] > 50:

        print (l1[x])
print(l1)

print(t1)
l1[2] = 99

print(l1)

# value of third position changed to 99
t1[2] = 99
# Functions of Tuple and list can be found using below

# print (dir(t1))

# print (dir(l1))
print(dir(l1))
# Dictionary (Associated with ('key' and 'values'))

dict = {"Empid": "1234", "EName": "Pavan", "Salary": 60000, "Email": "pavan@123.com"}

dict
dict['Empid']
def Add(x, y):

    return(x + y)
# Passing positional arguments - args

Add(2, 7) 
# Passing Named arguments/key-word arguments - kwargs

Add(y = 20, x = 30)
# We can also give default values to the arguments in function definition

def Add_any(x = 0, y = 0):

    return(x, y)
# Only positional arguments accepted, no named arguments allowed

def MyFunc(*args):

    print (args)
# function for addition of n arguments

def MyFunc1(*args):

    Sum = 0

    for i in args:

        Sum += i

    return (Sum)
MyFunc1(1,2,3)
# only named arguments accepted, no positional arguments allowed

def Myfunc2(**kwargs):

    print(kwargs)