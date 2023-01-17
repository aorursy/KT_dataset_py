my_list = ['A string',23,100.232,'o']
my_list[1:]
my_list + ['new item']
my_list * 2
my_dict = {'key1':'value1','key2':'value2'}
my_dict.keys()
my_dict.values()
my_dict.items()
t = (1,2,3)
list1 = [1,1,2,2,3,4,5,6,1,1]
set(list1)
loc = 'Bank'

if loc == 'Auto Shop':
    print('Welcome to the Auto Shop!')
elif loc == 'Bank':
    print('Welcome to the bank!')
else:
    print('Where are you?')
for num in list1:
    print(num)
for letter in 'This is a string.':
    print(letter)
list2 = [(2,4),(6,8),(10,12)]
#without unpacking
for tup in list2:
    print(tup)

#with unpacking
for (t1,t2) in list2:
    print(t1)    
d = {'k1':1,'k2':2,'k3':3}
# Dictionary unpacking
for k,v in d.items():
    print(k)
    print(v)
x = 0

while x < 10:
    print('x is currently: ',x)
    print(' x is still less than 10, adding 1 to x')
    x+=1
    
else:
    print('All Done!')
x = 0

while x < 10:
    print('x is currently: ',x)
    print(' x is still less than 10, adding 1 to x')
    x+=1
    if x==3:
        print('Breaking because x==3')
        break
    else:
        print('continuing...')
        continue
# Notice how 11 is not included, up to but not including 11, just like slice notation!
list(range(0,11))
list(range(0,101,10))
# Notice the tuple unpacking!
#enumerate helps tracking index
for i,letter in enumerate('abcde'):
    print("At index {} the letter is {}".format(i,letter))
list(enumerate('abcde'))
'x' in ['x','y','z']
from random import shuffle
mylist = [10,20,30,40,100]
min(mylist)
# This shuffles the list "in-place" meaning it won't return
# anything, instead it will effect the list passed
shuffle(mylist)
from random import randint
# Return random integer in range [a, b], including both end points.
randint(0,100)
input('Enter Something into this box: ')
[x**2 for x in range(0,11)]
[x for x in range(11) if x % 2 == 0]
# Convert Celsius to Fahrenheit
celsius = [0,10,20.1,34.5]
fahrenheit = [((9/5)*temp + 32) for temp in celsius ]
fahrenheit