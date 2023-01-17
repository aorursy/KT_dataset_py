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

nameCheck = input("What name would you like to check?")

for name in boys.keys():

    if name == nameCheck:

        # if it is 'john', get the value associated with john and use that value for the print statement

        # because the value is an integer, we have to cast it to a string in the print statement, with str().

        print("There were " + str(boys[name]) + " boys named " + name)

for name in boys.keys():

    if name in girls.keys():

       # print(name)

        pass

# commenting out to save space
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))

# 1. 

# Are there more boy names or girl names?

if len(boys) > len(girls):

    print("More boys than girl names")

elif len(boys) == len(girls):

    print("same amount of boys and girls")

elif len(boys) < len(girls):

    print("More girls than boy names")

else: 

    print("not working yet") # I could have made this the line above, but this I build in a validation



# What about for particular first letters? 

boyTally = 0

girlTally = 0

firstLetter = "q"

for name in boys.keys():

    if firstLetter in name[0]:

        boyTally += 1



for name in girls.keys():

    if firstLetter in name[0]:

        girlTally += 1



#print(str(boyTally) + " boys")

#print(girlTally + " boys")



if boyTally > girlTally:

    print("More boys with " + firstLetter + " names")

elif boyTally == girlTally:

    print("same amount of boys and girls with " + firstLetter +  " names")

elif boyTally < girlTally:

    print("More girls with " + firstLetter +  " names")

else: 

    print("not working yet") 





# What about for every first letter?

firstLetter = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]



for letter in firstLetter:

    boyCount = 0

    girlCount = 0 

    for name in boys.keys():

        if letter in name[0]:

            boyCount += 1

    for name in girls.keys():

        if letter in name[0]:

            girlCount += 1

    #print(str(girlCount) + " girls with " + letter + " names")

    #print(str(boyCount) + " boys with " + letter + " names")

    if boyCount > girlCount:

        print("More boys with " + letter + " names")

    elif boyCount == girlCount:

        print("same amount of boys and girls with " + firstLetter +  " names")

    else: 

        print("More girls with " + letter + " names")

              



### future improvements 

#gender = [boys, girls] --one loop, 2 times, figure out the tally

#alphabet = ["a":"z"] -- couldn't get this working yet, so I did it manually



# 2. How many babies are in the dataset (assuming nobody is counted more than once)?

genderNames = len(boys) + len(girls) 

print("There were %d boy and girl names." %genderNames)



countNames = 0

for line in open(NAMES_LIST, 'r').readlines():

    countNames += 1

print("There were " + str(countNames) + " records in this data.")



nonBinary = countNames - genderNames

if countNames == genderNames: 

    print("all babies had a M or F gender... is that a reliable dataset? %d non-binary babies?" %nonBinary)

else: 

    print("there's %d babies that weren't identified as M or F in this dataset" %nonBinary)
# 3. What is the longest name in the dataset?

leaderName = ""

longest = 0

leaderBoard = []



for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")

    #print(name)

    if len(name) > longest:

        leaderName = name

        longest = len(name)

        #print(longest)

        leaderBoard = []

    

    elif len(name) == longest:

        leaderBoard.append(name)

    

    elif len(name) < longest:

        pass

    

    else:

        print("not working yet")



print(leaderName + " is %d characters long and appears to be the longest name." %len(leaderName))

print("Along the way, the list grows of of names up to " + leaderName)

print(leaderBoard)
# 4. How many boy names are also girl names? How many girls' names are also boys' names?

# are there actually two questions in that? Isn't this just the intersection Q from before but by count?

overlap = 0

shared = []

for name in boys.keys():

    if name in girls.keys():

        overlap += 1

        shared.append(name) # this lets me see which names if I want to use them

print(overlap)

#print(shared)
# 5. How many names are subsets of other names?

# I can see how to make a variable to check for subsets, but struggling to pull out how to all possible subsets

# would love to have help on this in class tomorrow!

starter = input("what subset would you like to check?...")

subsetCount = 0



for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")

    #print(name)

    if starter in name:

        subsetCount += 1



print(str(subsetCount) + " names have " + starter + " in them")

# 6. What is the most popular girl name that is also a boy name?

uniqueNames = []

for name in shared:

    if name not in uniqueNames:

        uniqueNames.append(name)

    else:

        pass



print(len(shared))

print(len(uniqueNames))

# These shouldn't be the exact same amount - I'm confused on this one

# FAIL(sadface)
# 7. Write a program that will take a name as input and return the number of babies with that name in the girl and boy datasets.

def babyCount(nameCheck):

    counter = 0

    for name in boys.keys():

        if nameCheck == name:

            counter += 1

    for name in girls.keys():

        if nameCheck == name:

            counter += 1

    return(counter)

        



babyCount(input("enter a name... "))



# help me figure out what I'm missing!
# 8. Take a prefix as input and print the number of babies with that prefix in each dataset 

# (i.e., "m" would list babies whose names start with "m" and "ma" would list babies whose names start with "ma", etc).

prefix = input("enter a prefix...")

prefNames = []

prefixLength = len(prefix)

for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")

    if prefix in name[:prefixLength+1]: # I get nothing if I remove the +1, but confused by this and need help debugging

        prefNames.append(name)

        # print(name) -- for debugging, getting names with 1st letter different - not sure how to address

    else:

        pass

    

print(str(len(prefNames)) +" names start with "+ prefix + " as a prefix." )

print(prefNames)



# I'm feeling close but couldn't quite get this working
