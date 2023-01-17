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

if len(girls) > len(boys): #check total counts from above to compare

    print("There are more girls names than boys names in 2010.")

elif len(boys) > len(girls):

    print("There are more boys names than girls names in 2010.")

else:

    print("There is the same number of girls and boys names.")

    

from string import ascii_lowercase #import this to easily search through alphabet

for letter in ascii_lowercase: #run through each letter of the alphabet

    countBoys = 0     #assign a variable to count how many boys names start with each letter

    countGirls = 0     #assign a variable to count how many girls names start with each letter

    for name in boys.keys():    #lopp through all boys names

        firstLetter = name[0]    #extract the first letter of each name

        if letter in firstLetter:    #check if letter of alphabet matches first letter of name

            countBoys = countBoys +1    #if it does, add to total boys letter count

    for name in girls.keys():    #repeat for counting girls names

        firstLetter = name[0]

        if letter in firstLetter:

            countGirls = countGirls +1

    if countGirls > countBoys:    #compare girls and boys first letter totals

        print("There are more girls names that start with '"+letter.upper()+"'")

    elif countBoys > countGirls:

        print("There are more boys names that start with '"+letter.upper()+"'")

    else:

        print("There is the same number of boys and girls names that start with '"+letter.upper()+"'")

       # print(name + " " + str(boys[name]))

countBoys = 0

countGirls = 0

for num in boys: #loop through all the boys

    countBoys = countBoys + boys[num] #add the values together

for num in girls: #loop through all the girls

    countGirls = countGirls + girls[num] #add the values together

totalBabies = countBoys + countGirls  #add boy and girl totals together

print("There are " + f"{totalBabies:,}"+ " babies in the dataset.")
longestBoy = max(len(x) for x in boys) #Find longest name length in boys

longestGirl = max(len(x) for x in girls) #Find longest name length in girls

if longestBoy > longestGirl: #compare boys and girls to set longest length

    longestName = longestBoy

else:

    longestName = longestGirl

print("The longest names are:")

for name in boys: #loop through boys

    nameLen = len(name) 

    if nameLen == longestName: #check if length of current name is equal to the length of the longest name

        print(name)

for name in girls: #repeat for girls

    nameLen = len(name)

    if nameLen == longestName:

        print(name)

    
enterName = input("Enter a name: ") #Take a name

name = enterName.lower() #Change name to lowercase for easier matching

boyNum = 0 #Create variables in case no matches are found

girlNum = 0

if name in boys:  #check for name in boys

    boyNum = boys[name]  #get number value for that name

if name in girls:  #repeat for girls

    girlNum = girls[name]

print("There are " + f"{boyNum:,}"+" boys named " + enterName + ".") #print answer with number formatted as a string

print("There are " + f"{girlNum:,}" +" girls named " + enterName + ".")
NAMES_LIST_2013 = "/kaggle/input/a3data/yob2013.txt" #Copy code from above to import 2013 data



boys2013 = {}  # create an empty dictionary of key:value pairs for the boys

girls2013 = {} # create an empty dictionary of key:value pairs for the girls



for line in open(NAMES_LIST_2013, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    # Since 'count' is actaully a string of text and not an integer, 

    # we need to turn it into an integer to store that number in the dictionary so we can use it. 

    # later to do arithmetic that we couldn't do, if it was just text. This is called 'casting'.

    

    count = int(count)   # Cast the string 'count' to an integer

    

    if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

        girls2013[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys2013[name.lower()] = count

        

#my code below        

positiveGapHolder = 0 #create a place holder for cheking each name for highest increase gap

negativeGapHolder = 0 #create a place holder for cheking each name for highest decrease gap

positiveNameHolder = ""  #hold the matching name for each

negativeNameHolder = ""

for name in boys2013:  #loop through all 2013 boys names

    if name in boys:  #check for each name in 2010 names list

        lenNew = boys2013[name] #hold 2013 name popularity

        lenOld = boys[name]  #hold 2010 name popularity

        gap = lenNew - lenOld  #compute the gap in popularity between 2013 and 2010

        if gap > positiveGapHolder:  #compare this names gap to the holder for largest increase gap

            positiveGapHolder = gap  #if this name has a greater increase replace in the holder

            positiveNameHolder = name  #grab the eqivalent name

        if gap < negativeGapHolder:

            negativeGapHolder = gap

            negativeNameHolder = name

print("The name "+ positiveNameHolder.capitalize() + " increased the most in popularity for boys between 2010 and 2013.")

print("The name "+ negativeNameHolder.capitalize() + " decreased the most in popularity for boys between 2010 and 2013.")



#Repeat the process above for girls

positiveGapHolder = 0

negativeGapHolder = 0

positiveNameHolder = ""

negativeNameHolder = ""

for name in girls2013:

    if name in girls:

        lenNew = girls2013[name]

        lenOld = girls[name]

        gap = lenNew - lenOld

        if gap > positiveGapHolder:

            positiveGapHolder = gap

            positiveNameHolder = name

        if gap < negativeGapHolder:

            negativeGapHolder = gap

            negativeNameHolder = name

print("The name "+ positiveNameHolder.capitalize() + " increased the most in popularity for girls between 2010 and 2013.")

print("The name "+ negativeNameHolder.capitalize() + " decreased the most in popularity for girls between 2010 and 2013.")