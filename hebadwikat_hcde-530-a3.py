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

# Total number of names

boyNameCount = len(boys.keys())

girlNameCount = len(girls.keys())

if boyNameCount > girlNameCount:

    print('There are more boy names in total')

else:

    print('There are more girl names in total')

    

# particular first letter

# to test out a particular first letter , I used the .startwith to test a letter i entered, I looped in each dictionary and sat a counter for each , and then compared the two numbers from each dictionary.

firstLetter = 'h'

firstLetterBoyNameCount = 0

firstLetterGirlNameCount = 0



for name in boys.keys():

    if name.startswith(firstLetter):

        firstLetterBoyNameCount = firstLetterBoyNameCount+1

        

for name in girls.keys():

    if name.startswith(firstLetter):

        firstLetterGirlNameCount = firstLetterGirlNameCount+1



if firstLetterBoyNameCount > firstLetterGirlNameCount:

    print('There are more boy names for particular first letter: ' + firstLetter)

else:

    print('There are more girl names for particular first letter: ' + firstLetter)

    

# all first letters

#created new dictionaries to enter the first letter from girls and boys names in their dictionaries

boysFirstLetterDictionary = {}

girlsFirstLetterDictionary = {}



#we are looking for the first letter, so first we need to check if its in this new dictionary , if not then we count it and  add it through the else statement

for name in boys.keys():

    if name[0] in boysFirstLetterDictionary.keys():

        boysFirstLetterDictionary[name[0]] = boysFirstLetterDictionary[name[0]] + 1

    else:

        boysFirstLetterDictionary[name[0]] = 1

        

for name in girls.keys():

    if name[0] in girlsFirstLetterDictionary.keys():

        girlsFirstLetterDictionary[name[0]] = girlsFirstLetterDictionary[name[0]] + 1

    else:

        girlsFirstLetterDictionary[name[0]] = 1

# I believe there could be a better solution , like making an intersection set. but i couldnt figure it out completely

for letter in boysFirstLetterDictionary.keys():

    if boysFirstLetterDictionary[letter] > girlsFirstLetterDictionary[letter]:

        print('There are more boy names for this first letter: ' + letter)

    else:

        print('There are more girl names for this first letter: ' + letter)

# the counter will go through the both loops , in which it will count all the babies from both lists

babies = 0

for boy in boys.keys():

    babies = babies + boys[boy]

    

for girl in girls.keys():

    babies = babies + girls[girl]

    

print("There is {} babies".format(babies))
longestName = ''

#since the data is split in two dictionaries , I created a variable 'longestName', it will go into boys list and then girls list and keep comparing names till it finds the longestname

for boy in boys.keys():

    if len(longestName) < len(boy):

        longestName = boy

    

for girl in girls.keys():

    if len(longestName) < len(girl):

        longestName = girl

        

print("Longest Name is " + longestName)
#we are trying to find the intersection of baby names between the two dictionaries

# same concept of counter is being used here

boyNamesThatAreAlsoGirlNames = 0

for name in boys.keys():

    if name in girls.keys():

        boyNamesThatAreAlsoGirlNames = boyNamesThatAreAlsoGirlNames + 1

        

print("There are {} boy names that are also girl names".format(boyNamesThatAreAlsoGirlNames))

        

girlNamesThatAreAlsoBoyNames = 0

for name in girls.keys():

    if name in boys.keys():

        girlNamesThatAreAlsoBoyNames = girlNamesThatAreAlsoBoyNames + 1

        

print("There are {} girl names that are also boy names".format(girlNamesThatAreAlsoBoyNames))

        



 