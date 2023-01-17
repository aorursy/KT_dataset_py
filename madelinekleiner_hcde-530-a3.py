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

# get the number of boy names and the number of girl names

num_boys = len(boys.keys())

num_girls = len(girls.keys())



# print which gender has more names on the list

if num_boys > num_girls:

    print("There are more boy names than girl names")

elif num_girls > num_boys:

    print("There are more girl names than boy names")

else:

    print ("There are equal number of boy and girl names")

    

# get the number of boy names and the number of girl names that start with "m"



# initialize 2 new lists to keep track of m names

boymnames = []

girlmnames = []



# initilialize 2 variables for the count of m names

boymnames_count = 0

girlmnames_count = 0



# find all the boy names that start with m and put them in boymnames

for name in boys.keys():

    if name.startswith("m"):

        boymnames.append(name)



# count how long boymnames is

boymnames_count = len(boymnames)



# find all the girl names that start with m and put them in girlmnames

for name in girls.keys():

    if name.startswith("m"):

        girlmnames.append(name)

        

# count how long girlmnames is

girlmnames_count = len(girlmnames)



# sum the counts from girl and boy lists

totalmnames = boymnames_count + girlmnames_count



# print the result

print("There are " + str(totalmnames) + " names that start with 'm'")
# get the number of boy names and the number of girl names that start with any letter

# that the user inputs



# wait for the user to input any letter

letter = input("Type a letter and hit enter: ")



# initialize 2 new lists to keep track of names

boyLetterNames = []

girlLetterNames = []



# initilialize 2 variables for the count of names that begin with the user's letter

boyLetterNames_count = 0

girlLetterNames_count = 0



# find all the boy names that start with the user's letter and put them in a list

for name in boys.keys():

    if name.startswith(letter):

        boyLetterNames.append(name)



# count how long the boy list is

boyLetterNames_count = len(boyLetterNames)



# find all the girl names that start with the user's letter and put them in a list

for name in girls.keys():

    if name.startswith(letter):

        girlLetterNames.append(name)

        

# count how long the girl list is

girlLetterNames_count = len(girlLetterNames)



# sum the counts from girl and boy lists

totalLetterNames = boyLetterNames_count + girlLetterNames_count



# print the result

print("There are " + str(totalLetterNames) + " names that start with " + letter)
# iterate over both data sets and sum all the counts



# initilize variables for the number of babies

num_gbabies = 0

num_bbabies = 0

num_babies = 0



# iterate through girls.keys() and add up all the girls

for name in girls.keys():

    num_gbabies = girls[name] + num_gbabies

print("The total number of girl babies is " + str(num_gbabies))

    

# iterate through boys.keys() and add up all the boys

for name in boys.keys():

    num_bbabies = boys[name] + num_bbabies

print("The total number of boy babies is " + str(num_bbabies))



# sum the total number of boy and girl babies

num_babies = num_gbabies + num_bbabies



# print the result

print("The total number of babies in the dataset is " + str(num_babies))
# initialize list for "gender neutral" names

boynames_asgirlnames = []



# iterate through boys.keys() and find names that also appear in girls.keys()

for name in boys.keys():

    if name in girls.keys():

        boynames_asgirlnames.append(name)



# find the length of the gender neutral list

print("The number of gender neutral names is " + str(len(boynames_asgirlnames)))

    
# ask the user to input a name

name = input("Type a name and hit enter: ")



# iterate through boys.keys() to find a match for the user's input

for baby in boys.keys():

    if baby == name:

        print("The number of boys named " + name + " is " + str(boys[name]))

        

# iterate through girls.keys() to find a match for the user's input

for baby in girls.keys():

    if baby == name:

        print("The number of girls named " + name + " is " + str(girls[name]))

        

# i know this doesn't work well if there is no match

# need to figure out how to fix those other cases