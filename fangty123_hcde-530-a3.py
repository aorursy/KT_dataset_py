NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls



for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    # Since 'count' is actaully a string of text and not an integer, 

    # we need to turn it into an integer to store that number in the dictionary so we can use it. 

    # later to do arithmetic that we couldn't do, if it was just text. This is called 'casting'.

    

    count = int(count)   # Cast the string 'count' to an integer

    

    if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

        girls[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys[name.lower()] = count



# We need to format our text so that it can show both the text and an integer.

# But the print() function only takes strings and we have a string and an integer.

# To deal with this, we use the % sign to say what text goes where. In the example 

# below the %d indicates to put a decimal value in the middle of the sentence, and

# that decimal value - the length of 'girls' is indicated at the end of the function: len(girls)



print("There were %d girls' names." %len(girls))

print("There were %d boys' names." %len(boys))



# We did this for you at the end of the previous homework.

# It's a little weird to stuff numbers into sentences this way, but once you get 

# used to it, it's easy. You can do lots of other formatting like this.

# Here's an explanation of how it works: https://www.geeksforgeeks.org/python-output-formatting/
# iterate through the boys dictionary. for each key see if it is: 'john'

for name in boys.keys():

    if name == "john":

        # if it is 'john', get the value associated with john and use that value for the print statement

        # because the value is an integer, we have to cast it to a string in the print statement, with str().

        print("There were " + str(boys[name]) + " boys named " + name)

for name in boys.keys():

    if name in girls.keys():

        print(name)
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))

# compare the number of keys in boys and girls dictionary

more = 'boy' if len(boys) > len(girls) else 'girl' 

print('There are more %s names.' % more)



names_by_first_letter = {}  # create an new empty dictionary



for line in open(NAMES_LIST, 'r').readlines():

    # code copied from first example

    name, gender, count = line.strip().split(",")

    count = int(count)

    

    key = name.lower()[0]

    if key not in names_by_first_letter:

        # this first letter has not been created yet, initialize the dictionary

        names_by_first_letter[key] = {'F': 0, 'M': 0}



    if gender == "F":

        names_by_first_letter[key]['F'] += 1

    elif gender == "M":

        names_by_first_letter[key]['M'] += 1

        

for first_letter in  names_by_first_letter:

    print('For first letter {0}, there are {1} girls name and {2} boys name'.format(first_letter, names_by_first_letter[first_letter]['F'], names_by_first_letter[first_letter]['M']))

    
babies = 0  # keep a counter of babies

longest_length = 0 # keep a cache for longest_length of name

longest_name = '' # keep a cache for name of longest_length



for line in open(NAMES_LIST, 'r').readlines():

    # code copied from first example

    name, gender, count = line.strip().split(",")    

    babies += int(count)

    if len(name) > longest_length:

        longest_length = len(name)

        longest_name = name.lower()



print("There were %d babies in the dataset." % babies)

print("The longest name in the dataset is %s." % longest_name)
boys_name_in_girls = 0  # keep a counter of boys name in girls

girls_name_in_boys = 0  # keep a counter of girls name in boys



for line in open(NAMES_LIST, 'r').readlines():

    # code copied from first example

    name, gender, count = line.strip().split(",")    

    

    if gender == "F":

        if name.lower() in boys:

            girls_name_in_boys += 1

    elif gender == "M":

        if name.lower() in girls:

            boys_name_in_girls += 1



print("There are {0} boys name also girls name, and {1} girls name also boys name babies in the dataset.".format(boys_name_in_girls, girls_name_in_boys))

print("Using alternative solution (intersection) gives: %d" % len(set(boys.keys()).intersection(set(girls.keys()))))
result = 0 # keep a counter of names that are subset of others

current_index, compare_index = 0, 0



# combine the boys and girls name into one set to avoid going through twice

scanned_names = sorted(set(boys.keys()).union(set(girls.keys())), key=len)



# get the total number of names and keep track of progress

print('Total number of names: %d' % len(scanned_names))

progress_step = len(scanned_names) // 100

percent = 0

    

while current_index < len(scanned_names):

    compare_index = current_index + 1

    while compare_index < len(scanned_names):

        name, compare_name = scanned_names[current_index], scanned_names[compare_index]

        if name in compare_name and name != compare_name:

            result += 1

            break

        compare_index += 1

    current_index += 1

    if current_index > (progress_step * percent):

        print("%d percent execution finished" % percent)

        percent += 1



print("%d names are subsets of other names." % result)
result_name = ''  # keep the of most popular name so far

result_popularity = 0  # keep a counter of max popularity so far



for line in open(NAMES_LIST, 'r').readlines():

    # code copied from first example

    name, gender, count = line.strip().split(",")

    count = int(count)

    

    if gender == "F":

        if name.lower() in boys and count > result_popularity:

            result_popularity = count

            result_name = name



print("The most popular girl name that is also a boy name is {0}, and its popularity is {1}.".format(result_name, result_popularity))
input_name = input("Enter a name: ").lower()

boy_result, girl_result = '', ''  # initialize result message



if input_name in boys.keys():

    boy_result = "%d boys found" % boys[input_name]

if input_name in girls.keys():

    girl_result = "%d girls found" % girls[input_name]

    

print(';'.join([boy_result, girl_result]))
prefix = input("Enter a prefix: ").lower()

boy_result, girl_result = [], []



for name in boys.keys():

    if name.startswith(prefix):

        boy_result.append(name)

for name in girls.keys():

    if name.startswith(prefix):

        girl_result.append(name)

    

print('Found boys name:' + (';'.join(boy_result)))

print('Found girls name:' + (';'.join(girl_result)))
ALL_NAMES_LIST = ["/kaggle/input/a3data/yob2010.txt", "/kaggle/input/a3data/yob2011.txt", "/kaggle/input/a3data/yob2012.txt", "/kaggle/input/a3data/yob2013.txt"]



boy_name_popularity = {}  # dictionary to store boy name popularity across all 4 years

girl_name_popularity = {}  # dictionary to store girl name popularity across all 4 years

boy_result_name = ''  # keep the of most popular name so far for boys

boy_result_popularity = 0  # keep a counter of max popularity so far for boys

girl_result_name = ''  # keep the of most popular name so far for girls

girl_result_popularity = 0  # keep a counter of max popularity so far for girls



for file_name in ALL_NAMES_LIST:

    for line in open(file_name, 'r').readlines():

        # code copied from first example

        name, gender, count = line.strip().split(",")

        name = name.lower()

        count = int(count)

        

        if gender == "F":

            if name not in girl_name_popularity:

                girl_name_popularity[name] = 0

            girl_name_popularity[name] += count

        elif gender == "M":

            if name not in boy_name_popularity:

                boy_name_popularity[name] = 0

            boy_name_popularity[name] += count



for name in boy_name_popularity.keys():

    if boy_name_popularity[name] > boy_result_popularity:

        boy_result_popularity = boy_name_popularity[name]

        boy_result_name = name

print("The most popular boy name across 4 years is {0}, and its popularity is {1}.".format(boy_result_name, boy_result_popularity))



for name in girl_name_popularity.keys():

    if girl_name_popularity[name] > boy_result_popularity:

        girl_result_popularity = girl_name_popularity[name]

        girl_result_name = name

print("The most popular girl name across 4 years is {0}, and its popularity is {1}.".format(girl_result_name, girl_result_popularity))
NAMES_2010 = "/kaggle/input/a3data/yob2010.txt"

NAMES_2013 = "/kaggle/input/a3data/yob2013.txt"



# given a file name of a year, return dictionary to store boy and girl name popularity

def getPopularity(file_name):

    # dictionary to store boy/girl name popularity for current year

    boy_popularity, girl_popularity = {}, {} 

    

    for line in open(file_name, 'r').readlines():

        # code copied from first example

        name, gender, count = line.strip().split(",")

        name = name.lower()

        count = int(count)



        if gender == "F":

            boy_popularity[name] = count

        elif gender == "M":

            girl_popularity[name] = count

    return boy_popularity, girl_popularity



# given two dictionaries of 2010 and 2013, return the difference between dictionaries

def getDifference(popularity_2010, popularity_2013):

    difference = {}

    for name in popularity_2013.keys():

        if name in popularity_2010.keys():

            difference[name] = popularity_2013[name] - popularity_2010[name]

    return difference



# given the difference dictionary, return the name with most increment and decrement

def getMostDifference(difference):

    most_increment_name, most_decrement_name = '', '' 

    most_increment_popularity, most_decrement_popularity = 0, 0

    for name in difference.keys():

        if difference[name] > most_increment_popularity:

            most_increment_popularity = difference[name]

            most_increment_name = name

        elif difference[name] < most_decrement_popularity:

            most_decrement_popularity = difference[name]

            most_decrement_name = name

    return most_increment_name, most_decrement_name



# step 1. get the popularity of boy and girl name from 2 years

boy_popularity_2010, girl_popularity_2010 = getPopularity(NAMES_2010)

boy_popularity_2013, girl_popularity_2013 = getPopularity(NAMES_2013)



# step 2. get the difference

boy_difference = getDifference(boy_popularity_2010, boy_popularity_2013)

girl_difference = getDifference(girl_popularity_2010, girl_popularity_2013)



# step 3. find the most increment and decrement

boy_name_increment, boy_name_decrement = getMostDifference(boy_difference)

girl_name_increment, girl_name_decrement = getMostDifference(girl_difference)



print("The boy name with most increased popularity is {0}, and most declined popularity is {1}".format(boy_name_increment, boy_name_decrement))

print("The girl name with most increased popularity is {0}, and most declined popularity is {1}".format(girl_name_increment, girl_name_decrement))