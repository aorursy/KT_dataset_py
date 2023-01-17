print('Hello World!')
x = 2

y = 4

print(x+y)
#notation: INDENTATION MATTERS

def say_hello(recipient):

    print('Hello, ' + recipient)

say_hello('Jane')

say_hello('John')

#you can specify d

def say_hello(recipient='Default Name'):

    print('Hello, ' + recipient)

say_hello('Jane')
def double_me(some_input):

    the_double = some_input*2

    return the_double
result = double_me(10)

print(result)
# Assigning some numbers to different variables

num1 = 10

num2 = -3

num3 = 7.41

num4 = -.6

num5 = 7

num6 = 3

num7 = 11.11
# Addition

num1 + num2
# Subtraction

num2 - num3
# Multiplication

num3 * num4
# Division

num4 / num5
# Exponent

num5 ** num6
# Increment existing variable

print(num7)

num7 += 4

num7
# Decrement existing variable

num7 -= 2

num7
# Multiply & re-assign

num7 *= 5

num7
# Assign the value of an expression to a variable

num8 = num1 + num2 * num3

num8
# Are these two expressions equal to each other?

num1 + num2 == num5
# Are these two expressions not equal to each other?

print(num3)

print(num4)

num3 != num4
# Is the first expression less than the second expression?

num5 < num6
x = 10

if x < 0:

    print('You entered a negative integer')

elif x == 0:

    print('You entered 0')

elif x == 1:

    print('You entered the integer 1')

else:

    print('You entered an integer greater than 1')
mystring = 'I love python data science bootcamp'

if 'python' in mystring:

    print('yeahhhh python')
def exercise1(miles):

    #do something and then return the result!

    feet = miles*5280

    print("There are %d feet in %d miles" %(feet, miles))

    return feet

    
thenumber = exercise1(5)

print(thenumber)
def exercise2(age):

    #do something

    if age < 7: 

        return 'Have a glass of milk.'

    elif age<21:

        return 'Have a coke.'

    else:

        return 'Have a martini.'
store4later = exercise2(20)
exercise2(50)
# Assign some containers to different variables

list1 = [3, 5, 6, 3, 'dog', 'cat', False]

tuple1 = (3, 5, 6, 3, 'dog', 'cat', False)

set1 = {3, 5, 6, 3, 'dog', 'cat', False}

dict1 = {'name': 'Jane', 'age': 23, 'fav_foods': ['pizza', 'fruit', 'fish']}
# Items in the list object are stored in the order they were added

list1
# Items in the tuple object are stored in the order they were added

tuple1
# Items in the set object are not stored in the order they were added

# Also, notice that the value 3 only appears once in this set object

set1
# Items in the dict object are not stored in the order they were added

dict1['name']
#lists

mylist = [1, 4, 3, 2, 99, 100]
type(mylist)
mylist[3]
mylist[0:3]
mylist[::-2]
mylist[::-1] #reverse a list
mylist[::2] #every other element
# Measure some strings:

words = ['dog', 'pennsylvania', 'summer'] # a container (list) named words consisting of 3 strings

for  word in words:  # i is each string in words in indexed order; cat is 0, window is 1, defenestrate is 2

    print(word, len(word))  # print each string and the number of letters in each string
names = ['Sam', 'Erin', 'Jane', 'Thomas'] 

ages = [18, 21, 33]

for  name,age in zip(names,ages):

    print(name,'is',age,'years old') 
for i in range(5):  # from index 0 to 4

    print(i)
for myvar in mylist:

    print(myvar)
mylist.append(4)

print(mylist)

print(tuple1)
#tuples are immutabe

a = (1,2)

for i in a:

    a.append(1)
def exercise3(n):

    #do something

    num = 0

    for i in range(n+1):

        num += i

    return num
exercise3(14)
def exercise4(month,day,year):

    calendar = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

    m = calendar[month]

    print(str(m) +'/' +str(day) +'/'+ str(year))
exercise4('August',12, 2019)
def exercise5(appendfruit, replacefruit):

    fruitlist = ['apple','bannana','orange','mango','strawberry','peach']

    #do something
mynewfruit = 'durian'

myreplacefruit = 'papaya'

exercise5(mynewfruit,myreplacefruit)
# Assign a string to a variable

a_string = 'tHis is a sTriNg'
# Return an uppercase version of the string

a_string.upper()
# Return a lowercase version of the string

a_string.lower()
# Notice that the methods called have not actually modified the string

a_string
# Count number of occurences of a substring in the string

a_string.count('i')
# Count number of occurences of a substring in the string

a_string.count('is')
# Does the string start with 'this'?

a_string.startswith('this')
# Does the lowercase string start with 'this'?

a_string.lower().startswith('this')
# Return a version of the string with a substring replaced with something else

a_string.replace('is', 'XYZ')
# Split the string into individual strings (delimited by spaces)

a_string.split()
# Assign a list of common hotdog properties and ingredients to the variable hd

hd = [2, 'grilled', 'relish', 'chili']
# Add a single item to the list

hd.append('mustard')

hd
# Add multiple items to the list

hd.extend(['ketchup', 'mayo', 'onions'])

hd
# Remove a single item from the list

hd.remove('mayo')

hd
# Remove and return the last item in the list

hd.pop()

hd
# Remove and return an item at an index

hd.pop(4)

hd
# Assign keywords and their values to a dictionary called dict2

dict2 = {'name': 'Francie', 'age': 43, 'fav_foods': ['veggies', 'rice', 'pasta']}

dict2
# Add a single key-value pair fav_color purple to the dictionary dict2

dict2['fav_color'] = 'purple'

dict2
# Add all keys and values from another dict to dict2

dict3 = {'fav_animal': 'cat', 'phone': 'iPhone'}

dict2.update(dict3)

dict2
# Return a list of keys in the dict

dict2.keys()
# Return a list of values in the dict

dict2.values()
# Return a list of key-value pairs (tuples) in the dict 

dict2.items()
# Return a specific value of a key in dict2 that has multiple values

dict2['fav_foods'][2]  # veggies is 0, rice is 1, pasta is 2
# Remove key and return its value from dict2 (error if key not found)

dict2.pop('phone')

dict2
print(newvariable)
a=10

b=5

c=2

print(a/(a-b*c))
try:

    a=10

    b=5

    c=2

    print(a/(a-b*c))

except ZeroDivisionError:

    print('Whoops, you tried to divide by zero')
something_bad_is_happening = True

if something_bad_is_happening:

    raise ValueError("are you sure you want to be doing this?")
def exercise6(l,w,h):

    if l<=0 or w<=0 or h<=0:

        raise ValueError("All inputs must be positive.")

    else:

        A = (l+w)*h/2

    return A

    
exercise6(12.4,13.5,.01)
exercise6(14,-99,10)
#you've already learned for loops

squares = []             # creates an empty container (list) named "squares"

for x in range(10):      # for every number in the range from 0 to 9

    squares.append(x**2) # square that number and append it to the list called squares

squares                  # print the final form of the list



#joining strings from a list
#here is a more consice way of executing the same loop using list comprehension

squares = [1, 2, 3, 4, 5]

squares = [x**2 for x in range(10)]  # define a list whose entries are the numbers 0 to 9 squared

squares
# You can do the same thing with dictionaries for example

squares = {x: x**2 for x in (2, 4, 6)}

squares
list1 = [1,2,3]

list2 = ['a','b','c']

test = zip(list1,list2)

print(test.__next__())

print(test.__next__())
import numpy as np

import scipy

import matplotlib.pyplot as plt
#lets imagine you've done some internet research and 

#you find that there is a package called lmfit that solves your problem

import lmfit
#the exclamation point means you're actually running shell code, we wont worry about that for now.

!pip install lmfit 
#Trying again...

import lmfit
#numpy is almost always imported like so (for convenience)

import numpy as np
mylist = ['apple',16,'banana']

myarray = np.array(mylist)

myarray
[print(type(l)) for l in mylist]
def f1(x):

    return x**2 # x**2 is x"squared"



x = np.array([1,2,3,4]) #x data to integrate with

y = f1(x)

print(x)

print(y)

ages = np.array([15,92,2,19,33])

names = np.array(['Sally','Grandpa George','Baby Jack','Jane','Billy'])

print('Ages',np.sort(ages))

print('Indices',np.argsort(ages))

print('Names',names[np.argsort(ages)])

x = np.array([1,2,3,4,5,-1,-2,-3])

print(x<0)

x[x<0]

indices = np.array([0,1,2,3])

x[indices]
stringarray = np.array(['hello','my','name','is','aaron'])

test1 = np.array([0,1,2,3])

test = f1(test1)

print(test)
shakespeare = open('../input/shakespeare-plays/alllines.txt').readlines()
shakespeare
len(shakespeare)
shakespeare[1].strip()
shakespeare[1].lower().strip().split()
# It will be useful to define a function that cleans up any string you pass it.

def split_and_clean_line(mystring):

    #input: string

    

    #output: a python list of lowercase words without punctuation

    punc = '".!-:,?#$*'

    punc2 = "'"

    

    #your code here

    alist = mystring.lower().strip().split()

    

    outList =[]

    

    for word in alist:

        outWord = ''

        for letter in word:

            if (letter not in punc) and (letter not in punc2):

                outWord = outWord + letter

        outList.append(outWord)

                

    return outList

test = 'This i"s. a st\'r,.ing!!'

split_and_clean_line(test)
# I suggest you fill a dictionary with words ;)

counting_words = {}
# Your Code Here to fill the dictionary and count words

for s in shakespeare:

    line = split_and_clean_line(s)

    for word in line:

        if word in counting_words.keys():

            counting_words[word] += 1

        else:

            counting_words.update([(word, 1)])

            

print('This many different words!',len(counting_words.keys()))
#Now lets sort. 

sortedDict = sorted(counting_words.items(), key=lambda item: item[1], reverse=True)
print(sortedDict)
descendingValues = np.sort(np.array(list(counting_words.values())))[::-1]

descendingValueIndex = np.argsort(np.array(list(counting_words.values())))[::-1]

theKeys = np.array(list(counting_words.keys()))

sortedKeys = theKeys[descendingValueIndex]

print(sortedKeys[0:10])

print(descendingValues[0:10])
#Please print the 10 most frequent words and their respective counts

print(sortedDict[0:10])
#tweets = open('../input/election-day-tweets/election_day_tweets.csv').readlines()

tweets = open('../input/election-day-tweets-txt/election_day_tweets.txt').readlines()

print(len(tweets))
func(tweets)
x = np.arange(8)

x < 5
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)

B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)

A | B
x = np.linspace(0,1,20)

print('original x',x)

condition = (x > .3) & (x < .7)

print('boolean array',condition)

print('just what we want',x[condition])
class animal:



    def __init__(self,l,w,h,name):

        self.name = name #PROPERITES

        self.belly = []  #PROPERITES

        self.length = l

        self.width = w

        self.heigh = h

    

    def eat(self, otheranimal): #FUNCTIONS

        self.belly.append(otheranimal.name)

        print('Yummmm',otheranimal.name,'was delicious')
snakey = animal(10,1,1,'sankeyMcSnakes')

mickeymouse = animal(2,3,1,'Mickey')

buggsbunny = animal(5,5,5,'BuggsBunny')
print('Animal Name:',snakey.name)

print('Stomach:',snakey.belly)
snakey.eat(mickeymouse)

print(snakey.belly)

snakey.eat(buggsbunny)

print(snakey.belly)
data = np.genfromtxt('../input/c2numpy/Seattle2014.csv',names=True,delimiter=',')

print(data.dtype.names)

rainfall = data['PRCP']

inches = rainfall / 254.0  # 1/10mm -> inches

inches.shape
print(str(len(data['PRCP'][data['PRCP']==0])) +' days without rain')

print(str(len(data['PRCP'][data['PRCP']>0])) +' days with rain')

print(str(len(data['PRCP'][data['PRCP']>0.5])) +' days without rain more than 0.5 inches')

rainyDays = data['PRCP'][data['PRCP']>0]

print(str(np.median(rainyDays)) + ' inches median rainfall')

summerPRCP = data['PRCP'][np.where(np.logical_and(data['DATE']>20140621,data['DATE']<20140921))]

print(str(np.median(summerPRCP)) + ' median summer rainfall')

print(str(np.max(summerPRCP)) + ' maximum summer rainfall')

nonSummerPRCP = data['PRCP'][np.where(np.logical_or(data['DATE']<20140621,data['DATE']>20140921))]

print(str(np.median(nonSummerPRCP)) + ' median non-summer rainfall')

#print(rainyDays)
!ls ../input/sfpolice2016through2018

!head -10 ../input/sfpolice2016through2018/sf-police-incidents-short.csv

import numpy as np

data = np.genfromtxt('../input/sfpolice2016through2018/sf-police-incidents-short.csv',names=True,delimiter=',',dtype=(int,int,'U30','U30','U30','U30','U10')) #NEED TO SPECIFY THE DATA TYPE (NOT SMART UNFORTUNATELY...)

data.dtype.names
data['Category']
#new and improved code here