# Integer
i = 9
blah = 10
# Float

fl = 9.8
fl
i
blah
blah = blah + 1
blah
type(blah)
type(fl)
type(blah + 2)
type(blah + 2.2)
blah + 2.2
# comments start with a # and continue till the end of the line
# There is no "char" type. Just strings

name = 'Prasanna' # or you could also do    name = "Prasanna"
name
type(name)
# concatenating strings 

name + ' L S'
# repeating strings

name * 5
opinion = 'Weird seeing my name so many times'
opinion[0]
opinion[1]
# slicing
opinion[0:6] # starts from 0 and goes upto 4 (5 - 1)
opinion[:5]
opinion[5:]
opinion.split(' ')
# displays all the available methods of the current object
dir(opinion)
# type a method call such as opinion.split()
# Once we have parenthesis, place text cursor into the parenthesis and press Shift + Tab once to display function parameters
# Press Shift + Tab twice to see the function documentation
opinion.lower()
opinion = opinion.upper() + '!'
opinion
li = [1, 2, 3, 4, 5]
# slicing is same as with strings
li[0]
li[:3]
li[1:4]
li.append(6)
li
li.remove(6)
li
del li[0]
li
dir(li)
li.reverse()
li
li.sort()
li
names = ['Prasanna', 'Prajwal', 'Gaurav', 'Dolly', 'Manasa']
names[0]
names[2]
names[0][:4]
len(li)
li
li + ['str', 'blah blah']
contacts = {'Prasanna' : 21323123, 'Prajwal': 12123113}
contacts['Prasanna']
contacts['Prajwal']
contacts['Gaurav'] = 121
contacts['Gaurav']
contacts
del contacts['Gaurav']
contacts['Gaurav']
len(contacts)
li
if li[0] >= 1:
    print('Awesome')
else:
    print('Not so awesome')
li[0] = 1
if li[0] > 1:
    print('Extremely awesome')
elif li[0] == 1:
    print('Still kinda awesome')
else:
    print('Nope. Not awesome.')
li[1] =2 
if li[0] == 1 and li[1] == 2:
    print('I can see!')
if li[0] == 1 or li[1] == 3:
    print('Half vision is still pretty good, I guess.')
li
while li[0] < 4:
    li[0] += 1
    print(li)
li
for entry in li:
    print(entry)
len(li)
list(range(10))
for i in range(len(li)):
    print(li[i])
for i in range(10):
    print(i)
list(range(2, 11, 2))
for i in range(5, 12):  # similar to the slicing. Starts at 5 and goes till and including 9
    print(i)
for i in range(0, 10, 2):
    print(i)
for i in range(0, 10, 4):
    print(i)
for i in range(len(li)):
    print(li[i])
li
contacts.items()
for key, value in contacts.items():
    print(str(key) + ' --- ' + str(value))
contacts.keys()
contacts.items()
contacts.values()
## Side note, (1,2, 3) creates whats called a Tuple

tup = (1, 2, 3)
tup
dir(tup)
# no method to add itesm into the tuple. Tuples are immutable. Can't be changed once created.
tup
# cool thing about tuples is that they can be used in "tuple assignments" like so

first_val, second_val, third_val = tup
first_val
second_val
third_val
# This can be very handy as we'll see

# Btw, here's how to swap two variable values in python
a = 2
b = 3
print(str(a) + ' ' + str(b))
b, a = a, b
print(str(a) + ' ' + str(b))
a,b
tu = a,b
tu
b, a = tu
a, b
usersName = input('What is your name?')
usersName
# also works with int and floats
numIters = input('Enter the number of iterations')
numIters
type(numIters)
numIters = int(numIters)
numIters, type(numIters)
slope = input('Enter the slope for the line : ')
slope, type(slope)
slope = float(slope)
slope, type(slope)
def get_int(message):
    reply = input(message)
    reply = int(reply)
    return reply
numBhelPuris = get_int('How many Bhel Puris do you want to eat today?')
numBhelPuris, type(numBhelPuris)
def get_contact():
    name = input('Enter your name : ')
    number = input('Enter your number : ')
    return name, number  # creates a tuple and returns the tuple
name, number = get_contact()
name, number
class Contact():
    def __init__(self, name, number):  # __init__ is the constructor
        self.name = name
        self.number = number 
        # ******** NOTE ******** "name" variables scope ends after this function. 
        # But "self.name" can be accessed as long as the object is in memory(has a reference to it)
    
    def print_contact(self):
        print('Name : ' + self.name)
        print('Number : ' + self.number)
        

class ClassName():
    def doSomeStuff(self):
        print('stuff stuff')
con = Contact('Prasanna', '121133131')  # in general, ClassName() creates an object of that class
con.print_contact()
c = ClassName()
c.doSomeStuff()
con.name   # con.name is like telling self.name 
           #(Can sorta relate this to the "this" pointer in c++. self is like the "this" pointer)
con.number 
contact_list = []
num_contacts = 3
for i in range(num_contacts):
    name, number = get_contact()
    newCon = Contact(name, number)
    contact_list.append(newCon)
contact_list
for contact in contact_list:
    contact.print_contact()
    print()  # new line
# File IO
fHandle = open('fileName.txt', 'w')
fHandle.write('''hello
world
bye
running out of things to type
''')
fHandle.close()
fHandle = open('fileName.txt', 'r')
dir(fHandle)
fHandle.read()
fHandle.read()
fHandle.seek(0)
fHandle.read()
fHandle.seek(0)
fHandle.readline()
fHandle.readline()
fHandle.readline()
fHandle.readline()
fHandle.readline()
fHandle.seek(0)
for line in fHandle:
    print(line)
fHandle.close()
