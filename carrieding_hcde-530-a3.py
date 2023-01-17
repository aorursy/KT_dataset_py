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

NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



boys = {}

girls = {}



for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")    

    count = int(count)

    

    if gender == "F":

        girls[name.lower()] = count

    elif gender == "M":

        boys[name.lower()] = count

        

if len(girls) > len(boys):

    print("There were more girl names than boy names.")

elif len(boys) > len(girls):

    print("There were more boy names than girl names.")

else:

    print("There were equal numbers of boys names and girl names.")

    

# compare the numbers of names starting with "a" among boys and girls, zero index + "a"

# make an empty list for boy names and girl names

girlsname = []

boysname = []



for name in girls.keys():

    if name[0] == "a":

        girlsname.append(name)

for name in boys.keys():

    if name[0] == "a":

        boysname.append(name)

if len(boysname) > len(girlsname):

    print("There were more boy names than girl names starts with an a.")

elif len(girlsname) > len(boysname):

    print("There were more girl names than boy names starts with an a.")

else:

    print("There were equal numbers of boy names and girl names starts with an a.")

    
neutralboy = []

neutralgirl = []



for name in boys.keys():

    if name in girls.keys():

        neutralboy.append(name)

# print("There were " + str(len(neutralboy)) + " boys' names are also girls' names.")

print("There were %d boys' names are also girls' names." %len(neutralboy))



for name in girls.keys():

    if name in boys.keys():

        neutralgirl.append(name)

# print("There were " + str(len(neutralgirl)) + " girls' names are also girls' names.")

print("There were %d girls' names are also boys' names." %len(neutralgirl))
# create an empty list to store all the counts in

total = []



for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")    

    count = int(count)

    total.append(count)



print("There were %d babies born in 2010." %sum(total))
NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



boys = {}

girls = {}



for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")    

    count = int(count)

    

    if gender == "F":

        girls[name.lower()] = count

    elif gender == "M":

        boys[name.lower()] = count

        

babyname = input("Type a baby name (in lower case): ")



for name in girls.keys() or boys.keys():

    if name == babyname:

        print("There were %d girls with this name." %girls[name])

        print("There were %d boys with this name." %boys[name])

#    else:

#        print("There was no baby born in 2010 with this name.")