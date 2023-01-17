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

#defining variable 

totbbynmes = 0 

#adding the length of boys and girls dictionary to equal new variable

totbbynmes = len(boys) + len(girls)

#print out the number of babies in data set using print and  str function 

print("Total number of babies" + " " + "=" + " " + str(totbbynmes))





    

    
#defining variable

neutral_names = []

#for loop to search names in boys dictionary and if any names are also in girls dictionary 

#for every name in boys.keys 

for name in boys.keys():

    #if name is also in girls.keys

    if name in girls.keys():

        #then add the name to the neutral_names matrix using append function

        neutral_names.append(name)

#print out the number of boys' names that are also girls names 

print("%d boys' names that are also girls names." %len(neutral_names))



#exact code except girls and boys keys have been switched and variables in print function have also switched

for name in girls.keys():

    if name in boys.keys():

        neutral_names.append(name)

print("%d girls' names that are also boys' names." %len(neutral_names))
#asking person for what to search for

print("What prefix would you like to search?")

#variable for inputing whatever they want to find

prefix = input()

#empty matrix to populate in the for loop with all the names with the prefix 

prefix_names = []

#for loop for searching list one line at a time 

for line in open(NAME_LIST, 'r').readlines():

    #variable for each item on the lines in file NAME_LIST

    name, gender, count = line.strip().split(",")

    #the prefix iinputed is found in the name 

    if str(prefix) in name:

        #add that name to prefix matrix

        prefix_names.append(name)

        #print the names with that prefix

        print(name)

#print baby names with the prefix searched 

print("Total number of baby names with" + str(prefix) + "is %d" %len(prefix_names))


