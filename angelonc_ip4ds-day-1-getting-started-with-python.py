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

say_hello()
def double_me(some_input):

    the_double = some_input*2

    return the_double
result = double_me(10)

print(result)
x = 2

y = 4
x==3
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

print(num1 + num2)

num5
# Are these two expressions not equal to each other?

print(num3 != num4)

print(num3,num4)
# Is the first expression less than the second expression?

num5 < num6
x = -1

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
'python' in mystring
def exercise1(miles):

    nf = miles * 5280

    return 'There are ' + str(nf) + ' feet in ' + str(miles) + ' miles!'

    #do something and then return the result!
thenumber = exercise1(10)

print(thenumber)
def exercise2(age):

    if age < 7:

        out = 'Have a glass of milk.'

    elif age < 21:

        out = 'Have a coke.'

    else:

        out = 'Have a martini.'

    

    return out
out = exercise2(20)
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

dict1

dict1['name']
#lists

mylist = [1, 4, 3, 2, 99, 100]
type(mylist)
mylist[3]
mylist[0:3]
mylist[:3]
mylist[0:-2]
mylist[::-1] #reverse a list
mylist[::2] #every other element
# Measure some strings:

words = ['dog', 'pennsylvania', 'summer'] # a container (list) named words consisting of 3 strings

for  word in words:  # i is each string in words in indexed order; cat is 0, window is 1, defenestrate is 2

    print(word, len(word))  # print each string and the number of letters in each string
names = ['Sam', 'Erin', 'Jane'] 

ages = [18, 21, 33]

for  name,age in zip(names,ages): # what is zip?

    print(name,'is',age,'years old') 
for i in range(5):  # from index 0 to 4

    print(i)
for myvar in mylist:

    print(myvar)
mylist = [1,2,3]

mylist.append(4)

print(mylist)
#tuples are immutabe

a = (1,2)

for i in a:

    a.append(1)

def exercise3(n):

    print(sum(range(n+1)))

    #do something

exercise3(14)
def exercise3_alt(n):

    out = 0

    for i in range(n):

        out += (i+1)

    

    return out



exercise3_alt(14)
def exercise4(month,day,year):

    months = ('January','February','March','April','May','June','July','August','September','October','November','December')

    print(str(months.index(month)+1) + '/' + str(day) + '/' + str(year))

    #do something

exercise4('August',12, 2019)
def exercise4_alt(month,day,year):

    months = {'january':1,

              'february':2,

              'march':3,

              'april':4,

              'may':5,

              'june':6,

              'july':7,

              'august':8,

              'september':9,

              'october':10,

              'november':11,

              'december':12}

    print(str(months.lower(month)) + '/' + str(day) + '/' + str(year))

    #do something

exercise4('August',12, 2019)
# Iterate a dict

months = {'january':1,

              'february':2,

              'march':3,

              'april':4,

              'may':5,

              'june':6,

              'july':7,

              'august':8,

              'september':9,

              'october':10,

              'november':11,

              'december':12}

for key,value in months.items(): # months.items has two items (key and value), so we need two iterators to print each item

    print(key,value)
months['january']

#months.keys()

#months.values()
def exercise5(appendfruit, replacefruit):

    fruitlist = ['apple','bannana','orange','mango','strawberry','peach']

    print('First fruit: ' + str(fruitlist[0]))

    print('Last fruit: ' + str(fruitlist[-1]))

    print('3:5 fruits: ' + str(fruitlist[2:5]))

    print('First 3 fruits' + str(fruitlist[0:3]))

    

    if replacefruit:

        fruitlist[fruitlist.index('apple')] = replacefruit

    if appendfruit:

        fruitlist.append(appendfruit)

        

    print('New fruitlist: ' + str(fruitlist))

    

    

    #do something
fruitlist = ['apple','bannana','orange','mango','strawberry','peach']

fruitlist[-1]



fruitlist.index('apple')
mynewfruit = 'durian'

myreplacefruit = 'papaya'

exercise5(mynewfruit,myreplacefruit)
# add a single item

hd = [2,'grilled','relish','chili']

hd.append('ketchup')

print(hd)



# add multiple items

hd.extend(['ketchup','mayo','onion'])

print(hd)
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

    result = a/(a-b*c)

    #print(a/(a-b*c))

except ZeroDivisionError:

    result = 0

    print('Whoops, you tried to divide by zero')



print(result)
something_bad_is_happening = True

if something_bad_is_happening:

    raise ValueError("are you sure you want to be doing this?")
def exercise6(b1,b2,h):

    

    if (b1 <= 0) or (b2 <= 0) or (h <= 0):

        raise ValueError('One of the inputs is <= 0, try again')

    else:

        A = .5*(b1+b2)*h

        return(A)

        

        

    #do something
print(exercise6(15,5,10))

exercise6(12.4,13.5,.01)
exercise6(14,-99,10)
#you've already learned for loops

squares = []             # creates an empty container (list) named "squares"

for x in range(10):      # for every number in the range from 0 to 9

    squares.append(x**2) # square that number and append it to the list called squares

squares                  # print the final form of the list



#joining strings from a list
#here is a more consice way of executing the same loop using list comprehension

squares = [x**2 for x in range(10)]  # define a list whose entries are the numbers 0 to 9 squared

squares
# You can do the same thing with dictionaries for example

squares = {x: x**2 for x in (2, 4, 6)}

squares
list1 = [1,2,3,4]

list2 = ['a','b','c','d']

a = zip(list1,list2)



for x,y in a:

    print(x,y)
a = zip(list1,list2)

print(a.__next__())
print(a.__next__())
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

x = np.array([1,2,3,-3,-2,-1])

x[x<0]
# index using predefined index

I = np.array([0,1,2,3])

x[I]
str = np.array(['hello','my','name','is','chris'])

print([str=='chris'])
ages = np.array([15,92,2,19,33])

names = np.array(['Sally','Grandpa George','Baby Jack','Jane','Billy'])

print('Ages',np.sort(ages))

print('Indices',np.argsort(ages))

print('Names',names[np.argsort(ages)])

shakespeare = open('../input/shakespeare-plays/alllines.txt').readlines()
shakespeare
!ls ../input/shakespeare-plays/
len(shakespeare)
print(shakespeare[0])

print(shakespeare[1].strip())

shakespeare[0]
shakespeare[1].lower().strip().split()
# It will be useful to define a function that cleans up any string you pass it.

def split_and_clean_line(mystring):

    #input: string

    #output: a python list of lowercase words without punctuation

    

    a = mystring.lower().strip()

    

    # punctuation flags

    # punct = '!,.-_~`?@#$%^&*()[]{};:+\"'

    punct = ["'",'"','~','!','--','.',",",'-','?','`','@','#','$','%','^','&','(',')',"[","]",'{','}']

    

    # for each character, check against punctuation

    out = []

    for i in a:

        if i not in punct:

            out.append(i)

            

    # join the rest as a string

    alist = ''.join(out).split()

    

    return alist



print(split_and_clean_line(shakespeare[1]))

shakespeare[1]

words = {}



# for each line

for i in shakespeare:

    

    # split and clean that line, then for each word in the line:

    for w in split_and_clean_line(i):

        

        if w in words:

            # if in the dictionary already, iterate the count for that key

            words[w] += 1

        else:

            # if not, initialize the key for that word

            words[w] = 1



print(words)



#my_dict{line[1]:1}

#for i in line:

#    if i not in my_dict.keys():

#        my_dict(line[i]:1)
# now sort it

import operator

sorted_d = sorted(words.items(), key=operator.itemgetter(1), reverse=True)

print(sorted_d)
print('This many different words!',len(words.keys()))
#Please print the 10 most frequent words and their respective counts

sorted_d[0:10]
import operator

import numpy as np

import scipy

import matplotlib.pyplot as plt



def strip_shakespeare(fn):

    

    def split_and_clean_line(mystring):

        #input: string

        #output: a python list of lowercase words without punctuation



        a = mystring.lower().strip()



        # punctuation flags

        # punct = '!,.-_~`?@#$%^&*()[]{};:+\"'

        punct = ["'",'"','~','!','--','.',",",'-','?','`','@','#','$','%','^','&','(',')',"[","]",'{','}',':',';']



        # for each character, check against punctuation

        out = []

        for i in a:

            if i not in punct:

                out.append(i)

        

        # alternative

        # for p in punct:

        #     a = a.replace(p,'')

        # return a.lower().split()



        # join the rest as a string

        alist = ''.join(out).split()



        return alist

    

    shakespeare = open(fn).readlines()

    

    # dictionary

    words = {}



    # for each line

    for i in shakespeare:



        # split and clean that line, then for each word in the line:

        for w in split_and_clean_line(i):

            if len(w) > 5:

                if w in words:

                    # if in the dictionary already, iterate the count for that key

                    words[w] += 1

                else:

                    # if not, initialize the key for that word

                    words[w] = 1

                

    print('This many different words!',len(words.keys()))

    #sorted_d = sorted(words.items(), key=operator.itemgetter(1), reverse=True)

    #print(sorted_d[0:10])

    

    I = np.argsort(np.array(list(words.values())))[::-1]

    wList = np.array(list(words.keys()))[I]

    cList = np.array(list(words.values()))[I]

    print('-----')



    for i in range(10):

        print(wList[i],cList[i])

    

    #return words

    



# call fcn

fn = '../input/shakespeare-plays/alllines.txt'

strip_shakespeare(fn)

    

    
#tweets = open('../input/election-day-tweets/election_day_tweets.csv').readlines()

tweets = open('../input/election-day-tweets-txt/election_day_tweets.txt').readlines()

print(len(tweets))
fn = '../input/election-day-tweets-txt/election_day_tweets.txt'

strip_shakespeare(fn)
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
print(sum(data['PRCP']==0))

print(sum(data['PRCP']>0))

print(sum(data['PRCP']>.5))

print(np.median(data['PRCP'][data['PRCP']>0]))

plt.hist(data['PRCP'][data['PRCP']>0])
summer = []

for date in data['DATE']:

    month = int(np.array2string(date)[4:6])

    #day = int(np.array2string(date)[-1]

    summer.append((month >= 6) & (month <= 9))

print('ndays with no rain: ',sum(inches==0))

print('ndays with rain: ',sum(inches>0))

print('ndays with more than .5: ',sum(inches>.5))

print('median ran on rainy days: ',np.median(inches[inches>0]))

print('Max summer rain: ', np.max(inches[summer]))

print('Median summer rain: ',np.median(inches[summer]))

print('Median non-summer rain: ',np.median(inches[np.logical_not(summer)]))

plt.hist(inches[summer])

int(np.array2string(date)[-2])
!ls ../input/sfpolice2016through2018

!head -10 ../input/sfpolice2016through2018/sf-police-incidents-short.csv

import numpy as np

data = np.genfromtxt('../input/sfpolice2016through2018/sf-police-incidents-short.csv',names=True,delimiter=',',dtype=(int,int,'U30','U30','U30','U30','U10')) #NEED TO SPECIFY THE DATA TYPE (NOT SMART UNFORTUNATELY...)

data.dtype.names
data['Category']

np.shape(data)
from datetime import datetime

a = datetime.strptime(data['Date'][0],'%Y-%m-%dT%H:%M:%S.%f')
## incidents per year

years = []

date = []

for i in data['Date']:

    years.append(i[0:4])

    date.append(i[0:10])



years = np.array(years)

for year in np.unique(years):

    print('In ' + year + ' there were ' + str(sum(years == year)) + ' incedents')
## prct incidents per day

days = np.array(list(data['DayOfWeek']))

dayCount = []

for day in np.unique(days):

    dayCount.append(sum(days==day))

print(np.array(dayCount)/len(days))
## variability from week to week per day (need to find the number of events on a given day)

date = np.array(date)



#dates = np.unique(date[days == 'Friday'])



var = []

# for each weekday

for i in np.unique(days):

    

    print(i,len(dates))

    

    sumVar = []

    # for each unique date

    for j in np.unique(date[days == i]):

        

        # make a sum

        sumVar.append(sum(date == j))

        print(j,sum(date == j))

        

    var.append(np.std(sumVar))



    

        
print(var)
#new and improved code here