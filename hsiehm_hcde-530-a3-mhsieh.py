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

# Question 1 - Comparison of number of 
# More or Less
diff = len(boys) - len(girls) # get the numerical difference between the dictionary lengths, which should be the difference in unique names for boys and girls
if (diff > 1):
    print ("There are %d more boy names than girl names."%diff) # since the difference was positive, there are more boys
elif (diff < 1):
    print ("There are %d more girl names than boy names." %abs(diff)) # since the difference was negative, there are more girls; turn the difference into a positive number
else:
    print("There are an equal amount of girl and boy names.") # since neither condition was true, there must be the same number of boy and girl names
    
# First Letters

nameCountsB = {} # create empty dictionary to store keys for characters that begin boy names
nameCountsG = {} # create the same for girls

for name in boys: # go through all keys in the boy name dictionary
    nameCountsB[name[0]] = nameCountsB.get(name[0], 0) + 1 # create a key based on the first letter of the name, then add the value associated with the name to this new character key

for name in girls: # do the same for girl names
    nameCountsG[name[0]] = nameCountsG.get(name[0], 0) + 1

for letter in nameCountsB.keys(): # go through the keys 
    if letter in nameCountsG.keys(): # only compare if the letter starts out names for both genders
        diff2 = nameCountsB[letter] - nameCountsG[letter] # get the difference in values for each key
        if (diff2 > 1):
            print (f"There are {diff2} more boy names than girl names that begin with {letter}.") # print as earlier, but with easier formatting!
        elif (diff2 < 1):
            print (f"There are {abs(diff2)} more girl names than boy names that begin with {letter}.")
        else:
            print(f"There are an equal amount of girl and boy names that begin with {letter}.")


# Question 2 - Total babies used for dataset
totalValues = 0

for name in boys.keys(): # go through all the boy names and add values
    totalValues += boys[name]
for name2 in girls.keys(): # do the same for girls
    totalValues += girls[name2]
print(f"There are {totalValues} babies included in this dataset." )
# Question 3 - Longest Name
longestName =  "" # placeholder longest name
bnG = {**boys, **girls} # combine the dictionaries for the purpose of this problem, since we're not concerned with the actual values

for name in bnG.keys(): # go through all names
    if len(name) > len(longestName): # check if current name is longer than existing
        longestName = name # if it is, assign the current value as longest
    # don't need an else, since we don't care
        
print(f"Of all the baby names, {longestName.title()}, is the longest with len(longestName) characters.")


# Question 4 - gender neutral names
# isn't this just the example from earlier?

genderNeutralNames = 0 # counter variable for shared names

for name in boys.keys(): # go through boy names
    if name in girls.keys(): # check if that name exists for girls as well
        genderNeutralNames += 1 # if it does, that's another shared name

print(f"There are {genderNeutralNames} names that are given to both boys and girls.")
# Question 5 - subset names

numSubSets = 0

for name in bnG.keys(): # go through the combined list of names once
    for name2 in bnG.keys(): # check the names initially against every other name
        if name in name2:
            if name != name2: # if we're checking it against itself, ignore it
                numSubSets += 1
                break # since we've counted this name, we can check the next name, rather than looking at additional name2s
print(f"There are {numSubSets} names that are a subsets of another boy or girl name.")

        

            
# Question 6 - most popular girl name that is gender neutral
mPopValue = 0 # placeholder value
mPopName = "" # placeholder name

for name in girls.keys(): # go through the girl keys
    if name in boys.keys(): # check if girl name is also a boy name aka gender-neutral
        if(mPopValue < girls[name]): # check if current most popular name is less than current name being examined
            mPopName = name # assign the name as the new most popular name
            mPopValue = girls[name] # assign number as the new most popular value
        # we don't care about the else

print(f"The most popular girl name that is gender neutral is {mPopName.title()}, having {mPopValue} girls with the name.")
# Question 7 Write a program that will take a name as input and return the number of babies with that name in the girl and boy datasets.

def numBabies(babyName):
    totalB = 0 # keep track of boys and girls separately for additional information
    totalG = 0
    for name in boys.keys(): # go through boys dictionary; we can't just check for babyName immediately, in case they put a name that is a subset
        if name == babyName: # check if key matches input
            totalB = boys[name] # assign value to total number of boys
    print(f"There are {totalB} boys named {babyName}.") # print to give additional information
    
    # repeat steps for girls
    for name in girls.keys():
        if name == babyName:
            totalG = girls[name]
    print(f"There are {totalG} girls named {babyName}.")
    total = totalB + totalG # add the totals together to get a genderless total
    return total

bName = input("Please enter a baby name to search for: ") # ask for user input for a name
bNum = numBabies(bName) # call function
print(f"There are {bNum} babies in total named {bName}.") # print total
# Question 8 Take a prefix as input and print the number of babies with that prefix in each dataset 
# (i.e., "m" would list babies whose names start with "m" and "ma" would list babies whose names start with "ma", etc).

def subStrBabies(babyPre):
    pBaby = {} # create a placeholder map
    for name in boys.keys(): # look at every key in the boys list
        if name.startswith(babyPre): # checks that it starts with the user input
            pBaby[name] = pBaby.get(name, 0) + boys[name] # assign key and value
    # do the same for girls
    for name in girls.keys():
        if name.startswith(babyPre):
            pBaby[name] = pBaby.get(name, 0) + girls[name]
    return pBaby

bPrefix = input("Please enter the start of a baby name to search for: ") # ask for a prefix
babies = subStrBabies(bPrefix) # assign new dictionary based on function
total = 0 # placeholder to sum the number of names
for name in babies.keys():
    total += babies[name]    
print(f"There are {total} babies (both boys and girls) whose name begin with {bPrefix}")
print(f"The names are {babies.keys()}")


# Question 9 Which boy and girl names are the most popular across all four years in our dataset?

# modified code from Brock's existing example, comments for additions on my part
NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"
NAMES_LIST2 = "/kaggle/input/a3data/yob2011.txt"
NAMES_LIST3 = "/kaggle/input/a3data/yob2012.txt"
NAMES_LIST4 = "/kaggle/input/a3data/yob2013.txt"
files = [NAMES_LIST, NAMES_LIST2, NAMES_LIST3, NAMES_LIST4] # create a list for the files to iterate through

# placeholders for girl and boy popularity checks
gPopValue9 = 0 
gPopName9 = "" 
bPopValue9 = 0
bPopName9 = ""

boys9 = {}  # create an empty dictionary of key:value pairs for the boys
girls9 = {} # create an empty dictionary of key:value pairs for the girls

for x in files: # perform the same over the 4 files
    for line in open(x, 'r').readlines():  

        name, gender, count = line.strip().split(",")

        count = int(count)   

        if gender == "F":    
            girls9[name.lower()] = girls9.get(name.lower(), 0) + count # add current count to existing (if any)
        elif gender == "M": 
            boys9[name.lower()] = boys9.get(name.lower(), 0) + count

for name in girls9.keys(): # go through the girl keys
    if(gPopValue9 < girls9[name]): # check if current most popular name is less than current name being examined
        gPopName9 = name # assign the name as the new most popular name
        gPopValue9 = girls9[name] # assign number as the new most popular value
    # we don't care about the else

for name in boys9.keys(): # go through the boy keys
    if(bPopValue9 < boys9[name]): # check if current most popular name is less than current name being examined
        bPopName9 = name # assign the name as the new most popular name
        bPopValue9 = boys9[name] # assign number as the new most popular value
    # we don't care about the else

print(f"The most popular girl name over 2010-2013 is {gPopName9.title()}, with {gPopValue9} girls with the name.") # print out most popular girl name
print(f"The most popular boy name over 2010-2013 is {bPopName9.title()}, with {bPopValue9} boys with the name.") # print out most popular boy name
# Question 10 Which boy and girl names have increased most in popularity between 2010 and 2013? Which ones have declined most in popularity?
files = [NAMES_LIST, NAMES_LIST4]
 
diffG = {} # empty dictionary for difference over years for girls
diffB = {} # same for boys
 
# placeholders to keep track of the key/name and count differences between 2013 and 2010
bDiffNameLow = "" # placeholder for the greatest decrease in boy name
bDiffNameHigh = "" # placeholder for the greatest increase in boy name
bDiffLow = 0 # placeholder for corresponding value for greatest decrease
bDiffHigh = 0 # placeholder for corresponding vlaue for greatest increase
 
# similar placeholders for girls
gDiffNameLow = ""
gDiffNameHigh = ""
gDiffLow = 0
gDiffHigh = 0
 
# read and add names + counts to the dictionaries for 2010 data
for line in open(files[0], 'r').readlines():  
    name, gender, count = line.strip().split(",")
 
    count = int(count)
    
    # add to dictionaries
    if (gender == 'F'):
        diffG[name.lower()] = -1*count # since we'll be comparing change, start negative; accounts for names not in 2013
    else:
        diffB[name.lower()] = -1*count
 
# read in for the 2013 data
for line in open(files[1], 'r').readlines():  
    name, gender, count = line.strip().split(",")
 
    count = int(count)
   
    if (gender == 'F'):
        diffG[name.lower()] = count + diffG.get(name.lower(), 0) # add current count to existing (if any)
    else:
        diffB[name.lower()] = count + diffB.get(name.lower(), 0)

# check for increase/decrease and names for girls
for name in diffG.keys():
    if (diffG[name] >= 0): # check high values
        if (diffG[name.lower()] > gDiffHigh): # is it greater than our placeholder?
            gDiffNameHigh = name # if yes, assign current as our highest
            gDiffHigh = diffG[name.lower()] # also grab their name for reporting.
    else: # do the same for low values
        if (diffG[name.lower()] < gDiffLow):
            gDiffNameLow = name
            gDiffLow = diffG[name.lower()]

# do the same for boys
for name in diffB.keys():
    if (diffB[name.lower()] >= 0):
        if (diffB[name.lower()] > bDiffHigh):
            bDiffNameHigh = name
            bDiffHigh = diffB[name.lower()]
    else:
        if (diffB[name.lower()] < bDiffLow):
            bDiffNameLow = name
            bDiffLow = diffB[name.lower()]

# our lows are all negative, so reverse their sign so we can report out nicely.
gDiffLow *= -1
bDiffLow *= -1
 
print(f"The girl name that increased in popularity the most from 2010-2013 is {gDiffNameHigh.title()} with {gDiffHigh} more girls with the name in 2013 than 2010.")
print(f"The girl name that decreased in popularity the most from 2010-2013 is {gDiffNameLow.title()} with {gDiffLow} fewer girls with the name in 2013 than 2010.")
print(f"The boy name that increased in popularity the most from 2010-2013 is {bDiffNameHigh.title()} with {bDiffHigh} more boys with the name in 2013 than 2010.")
print(f"The boy name that decreased in popularity the most from 2010-2013 is {bDiffNameLow.title()} with {bDiffLow} fewer boys with the name in 2013 than 2010.")
