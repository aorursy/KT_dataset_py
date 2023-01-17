viking_song = "Spam " * 4
print(viking_song)
#abs(x):returns absolute value of int,float,complexnum
abs(-2),abs(3-4j)
#any(iterable):takes list,string,dictionary & returns True if at least one element is True else false
any(''),any('000')#empty iterator, 0(int) is false but '0'(string) is true
#all(iterable):if only all the elements of an iterable are true
all(''),all([1,2,3,4])
#append(list)
planets=['a','b']
planets.append('pluto')
planets
#ascii(object):It returns a string containing printable representation of an object.
ascii('Pythön is interesting')
#bin(num):converts an integer to binary
bin(5)
#bool(val):false if val is false or null, true if True
test = 'Easy string'
print(test,'is',bool(test))
#chr(integer from 0 through 1,114,111):returns a char whose unicode point is the integer
chr(87),chr(3201)
#complex([real[,imag]]):returns a complex number
complex(),complex(2,-3),complex('5-3j')
#delattr(obj,name):deletes name attribute from object obj
#del obj.name:does the same task
class Coordinate:
  x = 10
  y = -5
  z = 0

point1 = Coordinate()
print(point1.x,point1.y,point1.z)
delattr(Coordinate, 'z')
del Coordinate.y
#print(point1.x,point1.y,point1.z)
#uncomment above statement to see the change
#dict(keywarg)     #dict(iterable,keywarg)          #dict(mapping,keywarg)
dict(),dict(x=5, y=0) , dict([('x', 5), ('y', -5)]),dict({'x': 4, 'y': 5}, z=8)    
#dir(obj):returns a list of valid attributes
dir([1,2,3])
#divmod(x,y):returns a tuple of quotient & remainder
divmod(9,2)
#enumerate(iterable,start):
grocery = ['bread', 'milk', 'butter']

for item in enumerate(grocery):
  print(item)

print('\n')
for count, item in enumerate(grocery):
  print(count, item)

print('\n')
# changing default start value
for count, item in enumerate(grocery, 100):
  print(count, item)
#filter(func,iterable):filters the given iterable with help of the function
alphabets = ['a', 'b', 'd', 'e', 'i', 'j', 'o']

# function that filters vowels
def filterVowels(alphabet):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if(alphabet in vowels):
        return True
    else:
        return False

filteredVowels = filter(filterVowels, alphabets)

print('The filtered vowels are:')
for vowel in filteredVowels:
    print(vowel)
    
print('\n')    

#filter() method also works without the filter function
randomList = [1, 'a', 0, False, True, '0']

filteredList = filter(None, randomList)

print('The filtered elements are:')
for element in filteredList:
    print(element)
#format(value[,format_spec]): returns a formatted representation of a given value specified by the format specifier
format(1234, "*>+7,d")

#str.format()
planet='Pluto'
position=9
"{}, you'll always be the {}th planet to me.".format(planet, position)

pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390
#         2 decimal points   3 decimal points, format as percent     separate with commas
"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)
#help(object):returns with help LOL
help(dict)
#input([prompt]):reads a line from input and converts it into a string and returns it
#input('Enter a string:')
#iter(obj,sentinel):returns an iterator for the given obj
vowels = ['a', 'e', 'i', 'o', 'u']
v_iter=iter(vowels)
print(next(v_iter))
print(next(v_iter))
print(next(v_iter))
print(next(v_iter))
#list([iterable]):creates a list
print(list())
list('aeiou'),list(('a','e','i','o','u')),list(['a','b','c','d','e'])
#list(dictionary):creates a list with the keys of dictionary as items of the list. The order of the elements will be random when the keys are provided without values
list({'a', 'e', 'i', 'o', 'u'}),list({'a': 1, 'e': 2, 'i': 3, 'o':4, 'u':5})
#len(s):returns the number of items of an object
len([]),len(''),len(range(1,10)),len(b'Python')
#len(dictionaries,sets)
testSet = {1, 2, 3}
print(testSet, 'length is', len(testSet))

# Empty Set
testSet = set()
print(testSet, 'length is', len(testSet))

testDict = {1: 'one', 2: 'two'}
print(testDict, 'length is', len(testDict))

testDict = {}
print(testDict, 'length is', len(testDict))

testSet = {1, 2}
# frozenSet:elements can'tbe modified
frozenTestSet = frozenset(testSet)
print(frozenTestSet, 'length is', len(frozenTestSet))
#max(list):
print(max([3,2,5,6,7,8]))
print(max(["Python", "C Programming", "Java", "JavaScript"]))
#max(dictionaries):returns the largest key 
square = {2: 4, -3: 9, -1: 1, -2: 4}
print(max(square))
print(max(square,key=lambda k:square[k]))#returns key with largest value
#min():similiar as max
#map(function,iterable,...):applies a given function to each item of an iterable and returns a list of results
#The returned value from map() (map object) can then be passed to functions like list() (to create a list), set() (to create a set) and so on.
def calculateSquare(n):
    return n*n
numbers = num=(1, 2, 3, 4)
result = map(calculateSquare, numbers)
print(result)
print(set(result))
print(list(map(calculateSquare, num)))
#next(iteartor,default):
random = [5, 9, 'cat']
# converting the list to an iterator
random_iterator = iter(random)
print(random_iterator)
# Output: 5
print(next(random_iterator))
#pow(x,y,x):x raised to y modulo z
pow(2,4,3),pow(4,2)
#print(*objects, sep=' ',end='\n', file=sys.stdout, flush=False)
a = 5
print("a =", a, sep='00000', end='\n\n\n')
print("a =", a, sep='0', end='')
#print with file parameter
sourceFile = open('python.txt', 'w')
print('Pretty cool, huh!', file = sourceFile)
sourceFile.close()
#This program tries to open the python.txt in writing mode. If this file doesn't exist, python.txt file is created and opened in writing mode.
#Here, we have passed sourceFile file object to the file parameter. The string object 'Pretty cool, huh!' is printed to python.txt file (check it in your system).
#Finally, the file is closed using close() method.
#range(stop):Returns a sequence of numbers starting from 0 to stop-1
#range(start, stop[, step]):Returns a sequence of numbers starting from start to stop-1 with an increment of step
print(list(range(2,-14,-2)))
print(list(range(2,14,-2)))
#reversed(sequence)
seq_string = 'Python'
print(list(reversed(seq_string)))
#round(number,ndigits):return a float rounded to n_digits
print(round(10.7))
print(round(2.665, 2))
print(round(2.675, 2))
#Note: The behavior of round() for floats can be surprising. Notice round(2.675, 2) gives 2.67 instead of the expected 2.68. This is not a bug: it's a result of the fact that most decimal fractions can't be represented exactly as a float.
#When the decimal 2.675 is converted to a binary floating-point number, it's again replaced with a binary approximation, whose exact value is:
#2.67499999999999982236431605997495353221893310546875: due to this 2.675 is rounded to 2.67
#consider using the decimal module, which is designed for floating-point arithmetic
from decimal import Decimal

# normal float
num = 2.675
print(round(num, 2))

# using decimal.Decimal (passed float as string for precision)
num = Decimal('2.675')
print(round(num, 2))
#slice(start, stop, step)
py_string = 'Python'

# start = -1, stop = -4, step = -1
# contains indices -1, -2 and -3
slice_object = slice(-1, -4, -1)

print(py_string[slice_object])   # noh

#get sublist or subtuple
py_list = ['P', 'y', 't', 'h', 'o', 'n']
py_tuple = ('P', 'y', 't', 'h', 'o', 'n')

# contains indices 0, 1 and 2
slice_object = slice(3)
print(py_list[slice_object]) # ['P', 'y', 't']

# contains indices 1 and 3
slice_object = slice(1, 5, 2)
print(py_tuple[slice_object]) # ('y', 'h')    

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planets[0:3],planets[:3],planets[3:],planets[1:-1],planets[-3:]
#sorted(iterable, key=None, reverse=False):sorts the elements of a given iterable in a specific order
y_list = ['e', 'a', 'u', 'o', 'i']
print(sorted(py_list))

# string
py_string = 'Python'
print(sorted(py_string))

# vowels tuple
py_tuple = ('e', 'a', 'u', 'o', 'i')
print(sorted(py_tuple,reverse=True))

# dictionary
py_dict = {'e': 1, 'a': 2, 'u': 3, 'o': 4, 'i': 5}
print(sorted(py_dict, reverse=True))

#Notice that in all cases that a sorted list is returned.
#Note: A list also has the sort() method which performs the same way as sorted(). The only difference is that the sort() method doesn't return any value and changes the original list.

#Sort the list using sorted() having a key function:

# take the second element for sort
def take_second(elem):
    return elem[1]


# random list
random = [(2, 2), (3, 4), (4, 1), (1, 3)]

# sort list with key
sorted_list = sorted(random, key=take_second)

# print list
print('Sorted list:', sorted_list)
# Nested list of student's info in a Science Olympiad
# List elements: (Student's Name, Marks out of 100, Age)

participant_list = [
    ('Alison', 50, 18),
    ('Terence', 75, 12),
    ('David', 75, 20),
    ('Jimmy', 90, 22),
    ('John', 45, 12)
]
##Logic::
#(1,3) > (1, 4)
#False
#(1, 4) < (2,2)
#True
#(1, 4, 1) < (2, 1)
#True
# Nested list of student's info in a Science Olympiad
# List elements: (Student's Name, Marks out of 100 , Age)
participant_list = [
    ('Alison', 50, 18),
    ('Terence', 75, 12),
    ('David', 75, 20),
    ('Jimmy', 90, 22),
    ('John', 45, 12)
]


def sorter(item):
    # Since highest marks first, least error = most marks
    error = 100 - item[1]
    age = item[2]
    return (error, age)


sorted_list = sorted(participant_list, key=sorter)
print(sorted_list)
numbers = [2.5, 3, 4, -5]

# start parameter is not provided
numbers_sum = sum(numbers)
print(numbers_sum)

# start = 10
numbers_sum = sum(numbers, 10)
print(numbers_sum)
t1 = tuple()
print('t1 =', t1)

# creating a tuple from a list
t2 = tuple([1, 4, 6])
print('t2 =', t2)

# creating a tuple from a string
t1 = tuple('Python')
print('t1 =',t1)

# creating a tuple from a dictionary
t1 = tuple({1: 'one', 2: 'two'})
print('t1 =',t1)
def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    order = arrivals.index(name)
    return order >= len(arrivals) / 2 and order != len(arrivals) - 1
party_attendees = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
print(fashionably_late(party_attendees,'Gilbert'))
print(fashionably_late(party_attendees,'Ford'))
def word_search(documents, keyword):
    """
    Takes a list of documents (each document is a string) and a keyword. 
    Returns list of the index values into the original list for all documents 
    containing the keyword.

    Example:
    doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
    >>> word_search(doc_list, 'casino')
    >>> [0]
    """
    # list to hold the indices of matching documents
    indices = [] 
    # Iterate through the indices (i) and elements (doc) of documents
    for i, doc in enumerate(documents):
        # Split the string doc into a list of words (according to whitespace)
        tokens = doc.split()
        # Make a transformed list where we 'normalize' each word to facilitate matching.
        # Periods and commas are removed from the end of each word, and it's set to all lowercase.
        normalized = [token.rstrip('.,').lower() for token in tokens]
        # Is there a match? If so, update the list of matching indices.
        if keyword.lower() in normalized:
            indices.append(i)
    return indices               
doc_list = ["The Learn Python Challenge", "They bought a Casino. car", "Casinoville"]
word_search(doc_list, 'casino')
[
    {'name': 'Peach', 'items': ['green shell', 'banana', 'green shell',], 'finish': 3},
    {'name': 'Bowser', 'items': ['green shell',], 'finish': 1},
    # Sometimes the racer's name wasn't recorded
    {'name': None, 'items': ['mushroom',], 'finish': 2},
    {'name': 'Toad', 'items': ['green shell', 'mushroom'], 'finish': 1},
]
def best_items(racers):
    """Given a list of racer dictionaries, return a dictionary mapping items to the number
    of times those items were picked up by racers who finished in first place.
    """
    winner_item_counts = {}
    for i in range(len(racers)):
        # The i'th racer dictionary
        racer = racers[i]
        # We're only interested in racers who finished in first
        if racer['finish'] == 1:
            for i in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if i not in winner_item_counts:
                    winner_item_counts[i] = 0
                winner_item_counts[i] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
                i+1, len(racers), racer['name'])
                 )
    return winner_item_counts
sample = [
    {'name': 'Peach', 'items': ['green shell', 'banana', 'green shell',], 'finish': 3},
    {'name': 'Bowser', 'items': ['green shell',], 'finish': 1},
    {'name': None, 'items': ['mushroom',], 'finish': 2},
    {'name': 'Toad', 'items': ['green shell', 'mushroom'], 'finish': 1},
]
best_items(sample)
def hand_total(hand):
    
    total = 0
    # Count the number of aces and deal with how to apply them at the end.
    aces = 0
    for card in hand:
        if card in ['J', 'Q', 'K']:
            total += 10
        elif card == 'A':
            aces += 1
        else:
            # Convert number cards (e.g. '7') to ints
            total += int(card)
    # At this point, total is the sum of this hand's cards *not counting aces*.

    # Add aces, counting them as 1 for now. This is the smallest total we can make from this hand
    total += aces
    # "Upgrade" aces from 1 to 11 as long as it helps us get closer to 21
    # without busting
    while total + 10 <= 21 and aces > 0:
        # Upgrade an ace from 1 to 11
        total += 10
        aces -= 1
    return total

def blackjack_hand_greater_than(hand_1, hand_2):
    total_1 = hand_total(hand_1)
    total_2 = hand_total(hand_2)
    return total_1 <= 21 and (total_1 > total_2 or total_2 > 21)
    """
    Return True if hand_1 beats hand_2, and False otherwise.
    
    In order for hand_1 to beat hand_2 the following must be true:
    - The total of hand_1 must not exceed 21
    - The total of hand_1 must exceed the total of hand_2 OR hand_2's total must exceed 21
    
    Hands are represented as a list of cards. Each card is represented by a string.
    
    When adding up a hand's total, cards with numbers count for that many points. Face
    cards ('J', 'Q', and 'K') are worth 10 points. 'A' can count for 1 or 11.
    
    When determining a hand's total, you should try to count aces in the way that 
    maximizes the hand's total without going over 21. e.g. the total of ['A', 'A', '9'] is 21,
    the total of ['A', 'A', '9', '3'] is 14.
    
    Examples:
    >>> blackjack_hand_greater_than(['K'], ['3', '4'])
    True
    >>> blackjack_hand_greater_than(['K'], ['10'])
    False
    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])
    False
    """