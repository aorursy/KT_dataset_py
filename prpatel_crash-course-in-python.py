import this
# import module itself
import re
my_regex = re.compile("[0-9]+", re.I)
print(my_regex)

# import as an alias
import re as regex
my_regex = regex.compile("[0-9]+", regex.I)
print(my_regex)

# import specific values from a module
from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()
print(lookup, my_counter)

# (Not advised) Import the entire contents of a module
match = 8
from re import * # re has a match function
print(match)

# Delete a module
del re
print("Normal division:", 5/2) # works by default in Python 3, in Python 2 use from __future__ import division
print("Integer division:", 5//2)
def double(x):
    """This function doubles the input values.
    Also triple quotes allow multiline comments."""
    return x+x

# Pass function to other functions
def apply_to_one(f):
    """Calls the functions f with 1 as its argument"""
    return f(1)

my_double = double
x         = apply_to_one(my_double)
print("x equals:",x)

# Anomymous functions or lambdas
y = apply_to_one(lambda x: x+4)
print("y equals:",y)

another_double = lambda x: x*2    # Not recommended ?
def another_double(x): return 2*x # Recommended

# Default arguments
def my_print(message="default message"): print(message)
    
my_print("Not default message")
my_print()

# Arguments by name
def subtract(a=0,b=0): return a-b

print("9-5=",subtract(9,b=5))
print("5-0=",subtract(5))
print('You can have single quoted string')
print("I can have double quoted string")

# Encode special characters using "\"
this_is_a_tab = "\t"
print("You can't see a tab or John Cena:", this_is_a_tab)

# Create raw strings using r""
not_a_tab     = r"\t"
print("You're not fooling anyone:", not_a_tab)
try:
    print(0/0)
except ZeroDivisionError:
    print("ZeroByZero is a sin")
integer_list = [1,2,3]
heterogeneous_list = ["string", 0.1, True]
list_of_list = [integer_list, heterogeneous_list, []]
print("List length:", len(integer_list))
print("List sum:", sum(integer_list))

# Indexing
x = [2*i+3 for i in range(10)]
print("x is:",x)
print("zero =", x[0])
print("one =", x[1])
print("last =", x[-1])
print("secondLast =", x[-2])

# Slicing Lists
print("first_three =", x[:3])
print("four_to_end =", x[3:])
print("two_to_five =", x[1:5])
print("last_three =", x[-3:])

# Check for list membership. Checks all elements one by one so not recommended for long lists
print("Check for membership:", 9 in x)

# Concatenate lists: Extend
x = [1,2,3]
print("Old list:", x)
x.extend([4,5,6])
print("Extended list:", x)

# Concatenate lists: Add (to not modify original list)
x = [1,2,3]
print("Old list:", x)
y = x + [10,20,30]
print("List added to old list:", y)

# Append one item at a time
x = [1,2,3]
print("Old list:", x)
x.append(0)
print("Appended at the end:", x)

# Unpack lists
x, y = [10, 20]
print("x=",x, "y=",y)

try:
    x, y = [100, 200, 300]
except:
    print("Error: LHS should have same number of elements as RHS")

_, y = ["string", [0.1, True]]
print("y is now:", y)
this_is_a_tuple = (1,3)
also_a_tuple    = 4,5

try:
    this_is_a_tuple[1] = 2
except TypeError:
    print("Like I said, immutable")
def do_two_things(x,y): return x+y, x-y # Essentially also tuples
s, d = do_two_things(6,2)
print("s:",s,"d:",d,"s+d:",s+d)

def do_two_things_not_quite(x,y): return [x+y],[x-y]
s, d = do_two_things_not_quite(6,2)
print("s:",s,"d:",d,"s+d:",s+d)

x, y = 1, 10
y, x = x, y # Pythonic way to swap
empty_dict = {} # Pythonic
empty_dict = dict() # Less pythonic
grades = {"Metis":100, "Thebe":90}
print("grades:", grades)
thebe_grade = grades["Thebe"]
print("thebe grades:", thebe_grade)

# Keyerror for non existent keys
try:
    europa_grade = grades["Europa"]
except KeyError:
    print("No grade for Europa!")

# get method to avoid exception and return a default value
thebe_grade    = grades.get("Thebe", 0)
print("thebe grades:", thebe_grade)
amalthea_grade = grades.get("Amalthea", 0)
print("amalthea grades:", amalthea_grade)

no_ones_grade  = grades.get("no one") # default value is None
print("no ones grades:", no_ones_grade)

# Check for existence of a key
IO_has_grade    = "IO" in grades
print("IO has grade:", IO_has_grade)
metis_has_grade = "Metis" in grades
print("Metis has grade:", metis_has_grade)

# Assign key-value pairs
grades["Callisto"] = 80 # adds new entry
grades["Metis"]    = 75 # replaces old entry
print("grades:", grades)

# Example of structuring data using Dictionaries
tweet = {
    "user": "Adrastea",
    "text": "Moons of Jupiter",
    "retweet_count": 75,
    "hashtags": ["#Jupyter","#Python","#DataScience"]
}

print("tweet:", tweet)

# Query all items of a dictionary
tweet_keys    = tweet.keys()   # list of keys
tweet_values  = tweet.values() # list of values
tweet_items   = tweet.items()  # list of key-value pairs

print("slower:","user" in tweet_keys)
print("faster:","user" in tweet)

# defaultdict: to count words for example in a document
# defaultdict can be initialised with int, list, dict or a function depending on its use
from collections import defaultdict
word_count = defaultdict(int) # int initialises counter (value) of keys to 0
for word in tweet["hashtags"]:
    word_count[word] += 1

print("Word count:", word_count)

dd_list = defaultdict(list) # list() produces empty list as values of keys
dd_list[2].append(1)
print("dd_list:", dd_list)

dd_dict = defaultdict(dict) # dict() produces empty dicstionary as values of keys
dd_dict["IO"]["Mass"] = "89319e18"
print("dd_dict:", dd_dict)

dd_func = defaultdict(lambda: [0,0])
dd_func[2][1] = 1
print("dd_func:", dd_func)
from collections import Counter
c = Counter(["one", "tres", "two","two","tres", "uno", "tres"])
print("c:", c)

word_counts = Counter(tweet["hashtags"])
print(word_counts)

# print 2 most common words in c
for word, count in c.most_common(2):
    print(word, count)
s = set()
s.add(1)
s.add(4)
print("s:",s)
print(4 in s)

# in operation on sets is very fast so use sets for a large collection of items to test for membership
words_list = ["This", "is", "a", "long", "list"] # using "in" to check membership is slow 
words_set  =  set(words_list) # very fast to check in set

# Sets can also be used to find unique entries
item_list = [1,2,3,1,2,3]
item_set  = set(item_list)
print("full list:", item_list, "\t unique entries:", list(item_set))
# if
if 1>2:
    print("In another universe")
elif 1>3:
    print("Yet another universe")
else:
    print("Here ruleth Math")

x = 2
# If else in 1 line
parity = "even" if x%2 == 0 else "odd"
print(parity)

# while
x = 0
while x < 5:
    print(x, "is less than 5")
    x += 1

# this is useful
for x in range(5):
    if x == 3:
        continue # go immediately to next iteration
    if x == 5:
        break # quit the loop
    print(x)
one_is_less_than_two = 1<2
true_equals_false    = True == False

# Following are all equivalent to boolean False
flist = [False, None, [], {}, "", set(), 0, 0.0]

for f in flist:
    if f:
        print("Not equivalent to Boolean False")
    else:
        print(f,"is equivalent to False")
        
all_function = all([1,True, 0.5, {"positive"}])
print("all:", all_function)
any_function = any(flist)
print("any:", any_function)
x = [4,1,2,3]
# sort and return a new list
y = sorted(x)
# sort the original list itself
x.sort()
print(y,x)

# Sort based on result of a function (specified by key) and in reverse
x = sorted([-4,1,-2,3], key=abs, reverse=True)
even_numbers = [x for x in range(10) if x%2 == 0]
print("list:", even_numbers)

# Turn lists into dictionaries or sets
square_dict = {x:x*x for x in even_numbers}
print("dict:", square_dict)

square_set  = {x*x for x in even_numbers}
print("set:",square_set)

zeros         = [0 for _ in even_numbers]
list_of_pairs = [(x,y) for x in range(2) for y in range(3)] # 2x3 pairs
print("list of pairs:", list_of_pairs)

increasing_pairs = [(x,y) for x in range(3) for y in range(x+1,5)]
print("increasing pairs:", increasing_pairs)
def lazy_range(n):
    """a lazy version of range"""
    i = 0
    while i < n:
        yield i
        i+=1

for i in lazy_range(5):
    j = i*2
import random
four_uniform_random = [random.random() for _ in range(4)]
print("four randoms:", four_uniform_random)

# Seed
random.seed(10)
print("With seed 10:",random.random())
random.seed(10)    
print("Re-initialised seed:",random.random())

# Randomise seed for examples below
import time
random.seed(time.time())

# Range
random_in_range = [random.randrange(30,60) for _ in range(4)]
print(random_in_range)

# Re-order elements of a List
up_to_ten = list(range(10))
random.shuffle(up_to_ten)
print("Shuffled list:",up_to_ten)

# Randomly pick one element from a List
random_choice = random.choice(up_to_ten)
print("Choice out of List:",random_choice)

# Sample without replacement
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
print("Samples without replacement:",winning_numbers)

# Sample with replacement (allowing duplicates)
choice_with_replacement = [random.choice(up_to_ten) for _ in range(6)]
print("Samples with replacement:", choice_with_replacement)
class Set:
    # these are member functions
    # each takes a first parameter "self" (another convention) that refers to the particular "Set" object being used
    
    def __init__(self, values=None):
        """ This is the constructor. It gets called when a new Set is created.
        It is used as:
        s1 = Set() # empty set
        s2 = Set([1,2,3,4]) # initialise with values """
        
        self.dict = {} # Each instance of Set has its own dict property
                       # which is what we'll use to track memberships
        if values is not None:
            for value in values:
                self.add(value)
                
    def __repr__(self):
        """ This is the string representation of a Set object if you type it at the Python prompt
         or pass it to str() """
        return "Set: " + str(self.dict.keys())
    
    # Well represent membership by being a key in self.dict with value True
    def add(self, value):
        self.dict[value] = True
        
    # Value is in the Set if it's a key in the dictionary
    def contains(self, value):
        return value in self.dict
    
    def remove(self, value):
        del self.dict[value]
        
s = Set([1,2,3,4])
s.add(5)
print(s)
print(s.contains(3))
s.remove(2)
print(s.contains(2))
def int_sum(v,w):
    return v+w

def int_sum_with_parameter(v,w,param1=1,param2=2):
    return v*param1+w*param2

# Map
# Let's say we'd like to add 2 vectors using above function
vector1 = [1,2,3]
vector2 = [3,2,1]

vector_sum = [v_i+w_i for v_i,w_i in zip(vector1,vector2)]
print(vector_sum)

map_vector_sum = list(map(int_sum,vector1,vector2))
print(map_vector_sum)

map_vector_sum_with_parameter = list(map(lambda x,y: int_sum_with_parameter(x,y,param2=1),vector1,vector2))
print(map_vector_sum_with_parameter)

# Reduce
from functools import reduce
"""combines the first two elements of a list, then that result with the third,
    that result with fourth and so on, producing a single result"""

def vector_add(v,w):
    """adds corresponding elements"""
    return [v_i+w_i for v_i, w_i in zip(v,w)]

# Let's say we want to add 4 vectors using this function
vectors = [[1,2,3],[4,5,6],[3,2,1],[6,5,4]]

vectors_add = list(reduce(vector_add,vectors)) 
print(vectors_add)
# Not pythonic
le_random_list = ["hihihi", "python", "ain't", "got", "shit", "on", "me"]
word           = []

for i in range(len(le_random_list)):
    word = word+[le_random_list[i]]
    
# Pythonic
word_pythonic = []
for i, element in enumerate(le_random_list):
    word_pythonic = word_pythonic + [element]
    
# Just indices
word_indexed = []
for index, _ in enumerate(le_random_list):
    word_indexed = word_indexed + [le_random_list[index]]
list1 = [1,2,3]
list2 = ["a","b","c"]
pairs = list(zip(list1, list2))
print("pairs:", pairs)

# Unzip: asterisk performs argument unpacking treating elements of pairs as individual arguments to zip
numbers, letters = zip(*pairs)
print("numbers:", numbers, "\tletters:", letters)

# Using with a function
def add(x,y): return x+y
add(*[1,2])
def doubler(f):
    """Take a function as an input and gives another function which will give double of the original function"""
    def g(x): # defined to take only 1 input here
        return 2*f(x)
    return g

def f1(x): return x+1
g = doubler(f1)
print("Doubler for f1(3):", g(3))

# However since g is defined to take only 1 input the following will not work
def f2(x,y): return x+y+1
g = doubler(f2)
try: g(1,2)
except TypeError:
    print("TypeError because 2 inputs given")

# Function definition with unnamed arguments "args" and unnamed keywords "kwargs"
def magic(*args, **kwargs):
    print("unnamed arguments:", args)
    print("unnamed keywords:", kwargs)

magic(1,2, key="word", key2="newWord")

def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        return 2*f(*args, **kwargs)
    return g
g = doubler_correct(f2)
print("Doubler for f2(1,2):", g(1,2))