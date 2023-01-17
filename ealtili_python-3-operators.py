range(0,11)
# Notice how 11 is not included, up to but not including 11, just like slice notation!
list(range(0,11))
list(range(0,12))
# Third parameter is step size!
# step size just means how big of a jump/leap/step you 
# take from the starting number to get to the next number.

list(range(0,11,2))
list(range(0,101,10))
index_count = 0

for letter in 'abcde':
    print("At index {} the letter is {}".format(index_count,letter))
    index_count += 1
# Notice the tuple unpacking!

for i,letter in enumerate('abcde'):
    print("At index {} the letter is {}".format(i,letter))
list(enumerate('abcde'))
mylist1 = [1,2,3,4,5]
mylist2 = ['a','b','c','d','e']
# This one is also a generator! We will explain this later, but for now let's transform it to a list
zip(mylist1,mylist2)
list(zip(mylist1,mylist2))
for item1, item2 in zip(mylist1,mylist2):
    print('For this tuple, first item was {} and second item was {}'.format(item1,item2))
'x' in ['x','y','z']
'x' in [1,2,3]
mylist = [10,20,30,40,100]
min(mylist)
max(mylist)
from random import shuffle
# This shuffles the list "in-place" meaning it won't return
# anything, instead it will effect the list passed
shuffle(mylist)
mylist
from random import randint
# Return random integer in range [a, b], including both end points.
randint(0,100)
# Return random integer in range [a, b], including both end points.
randint(0,100)
input('Enter Something into this box: ')
# Less Than
1 < 2
# Great Than
1 > 2
# Check for equality
1 == 1
# Check for inequality
1 != 1
# Less than or equal to
1 <= 3
# Greater than of equal to
1 >= 1
1 == 1 and 2 == 2
1 == 1 and 2 ==1
1 == 1 or 2 == 10
not 1 == 1
# Not very common to see something like this, but it still works
1 == 1 and not 2 == 3
# A more realistic example
answer = 'no'

if 1 == 1 and not answer == 'yes':
    print("Success!")
# Same as
answer = 'no'
if 1 ==1 and answer != 'yes':
    print("Success!")