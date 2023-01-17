NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls



for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    # Since 'count' is actually a string of text and not an integer, 

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

# Exercise 1. Are there more boy names or girl names? What about for particular first letters? What about for every first letter?

#Compare length of boys versus girls

if len(girls) > len(boys):

    print("There are more girl names than boy names.")

elif len(boys) > len(girls):

    print("There are more boy names than girl names.")

else:

    print("There are an equal number of girl names and boy names.")

    

#What about particular first letters?

#Defining a reusable procedure with an input parameter

def firstLetterCompare(letter):

    #Finding the number of occurrences in boy names of the parameter being the first letter

    countBoys = 0

    for name in boys.keys():

        if letter in name[0]:

            countBoys = countBoys + 1

            

    #Finding the number of occurrences in girl names of the parameter being the first letter

    countGirls = 0

    for name in girls.keys():

        if letter in name[0]:

            countGirls = countGirls + 1

    

    #Comparing the number of names starting with the parameter for girl names and boy names

    if countGirls > countBoys:

        print("There are more girls' names starting with letter '" + letter + "' than boys.")

    elif countBoys > countGirls:

        print("There are more boys' names starting with letter '" + letter + "' than girls.")

    else:

        print("There are the same number of girls and boys names starting with letter '" + letter + "'.")



#Calling the procedure for letter 'a'

firstLetterCompare('a')





#What about for every first letter?

#Creating a list of all possible letters

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#Iterating through each letter of the alphabet in the list

for alpha in alphabet:

    #Using the previous defined procedure

    firstLetterCompare(alpha)

            

            
#Exercise 2. How many babies are in the dataset (assuming nobody is counted more than once)?

#Calculating the sum of all counts for boy names and girl names

sum = len(boys) + len(girls)

print('There are ' + str(sum) + ' babies in the dataset.')
#Exercise 3. What is the longest name in the dataset?

longestName = ""

longestNameList = []



#Iterating through boys names to determine which name is the longest

for name in boys.keys():

    if len(name) > len(longestName):

        longestName = name



#Iterating through girls names to determine which name is longer than the longest in the boys names

for name in girls.keys():

    if len(name) > len(longestName):

        longestName = name



#Handling the equal length situation to show all boys and girls names that are of equal length to the longest

for name in boys.keys():

    if len(name) == len(longestName):

        longestNameList.append(name)

    

for name in girls.keys():

    if len(name) == len(longestName):

        longestNameList.append(name)



print("The longest name(s) in the dataset are:")

print(longestNameList)

        
#Exercise 4. How many boy names are also girl names? How many girls' names are also boys' names?

#Iterating through the boy names 

boyNameCount = 0

for boyName in boys.keys():

    #Evaluating whether boy name is found in girl names

    if boyName in girls.keys():

        boyNameCount = boyNameCount + 1

        

print("There are " + str(boyNameCount) + " boy names that are also girl names.")



#Iterating through the girl names

girlNameCount = 0

for girlName in girls.keys():

    #Evaluating whether girl name is found in boy names

    if girlName in boys.keys():

        girlNameCount = girlNameCount + 1

        

print("There are " + str(girlNameCount) + " girl names that are also boy names.")
#Exercise 7. Write a program that will take a name as input and return the number of babies with that name in the girl and boy datasets.



#Creating a reusable program to return the number of babies with a specified name in girl and boy datasets 

def boysAndGirlsTotal(inputName):

    #Creating the input prompt, depending on whether you want to run as a program or with input field

    #print("Enter a lowercase name > ")

    #inputName = input('Enter a name > ')

    boy_count = 0

    girl_count = 0

    

    #Iterating through boys names to see if input name is found

    for name in boys.keys():

        #If name is found, then print the count of boys names

        if name == inputName:

            boy_count = boys[name]

            print("There are " + str(boy_count) + " boys named " + name)

    

    #Prompt for when the name is not found in boys names

    if boy_count == 0:

        print("There are 0 boys named " + inputName)

    

    #Iterating through girls names to see if input name is found

    for name in girls.keys():

        #If name is found, then print the count of girls names

        if name == inputName:

            girl_count = girls[name]

            print("There are " + str(girl_count) + " girls named " + name)



        #Prompt for when the name is not found in girls names

    if girl_count == 0:

        print("There are 0 girls named " + inputName)

        

boysAndGirlsTotal('kyle')

boysAndGirlsTotal('aaron')

boysAndGirlsTotal('holly')