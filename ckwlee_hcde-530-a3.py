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



#Compare the amount of boys and girls name. Prints the gender who has the most and the difference.

if len(girls) == len(boys):

    print("There were equal amount of boys and girls names.")

elif len(girls) > len(boys):

    difference = len(girls) - len(boys)

    print("There were %d more girls' names than boy names." %difference)

else: 

    difference = len(boys) - len(girls)

    print("There were %d more boys' names than girl names." %difference)

    
NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



babies = 0 #create empty variable for counting number of babies



for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    # Since 'count' is actaully a string of text and not an integer, 

    # we need to turn it into an integer to store that number in the dictionary so we can use it. 

    # later to do arithmetic that we couldn't do, if it was just text. This is called 'casting'.

    

    count = int(count)   # Cast the string 'count' to an integer

    

    babies = babies + count #add count to current count of number of babies



#print final count of babies in dataset

print("There are %d number of babies in this dataset." %babies)

NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"

longest_name = "" #create empty string to store the longest name



for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")

    

    #replace stored name when this name is longer than the currently store longest name.    

    if len(name) > len(longest_name):

        longest_name = name

        



#print longest name and its length

print("The longest name is " + longest_name + " with %d letters." %len(longest_name))

    





NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"

boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls

girlcount = 0 #create empty count of girl names

boycount = 0 #create empty count of boy names



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

        

#loop through all the boy names and count how many girl names are present, print final count

for name in boys.keys():

    if name in girls.keys():

        girlcount = girlcount + 1



print("There are " + str(girlcount) + " girls' names that are boys' names")



#loop through all the girl names and count how many boy names are present, print final count

for name in girls.keys():

    if name in boys.keys():

        boycount = boycount + 1      



print("There are " + str(boycount) + " boys' names that are girls' names")
NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



all_names = {}  # create an empty dictionary of key:value pairs for all the names in the file

subset_count = 0 # create variable to keep track of count names with subsets



for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")

    

    all_names[name.lower()] = 0; #add name to dictionary with empty value for counting number of subset of name



for name in all_names.keys(): #loop through all names

    check_name = name #store current name to be used a comparison

    for name in all_names.keys(): #loop through all names to find subsets of names

        if check_name in name:

            if check_name != name: #does not count if subset is a subset of itself

                all_names[check_name] = all_names[check_name] + 1 # keeps count of number of subsets

    if all_names[check_name] > 0: #counts the number of names that are subsets

        subset_count = subset_count + 1

        #Test code to print name, subset of name and running count of total subsets:

        #print(check_name +" %d" %all_names[check_name])

        #print(subset_count)



#Print final count of names with subsets

print("There are %d names that are subsets of names." %subset_count)





        



NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"

boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls

popular = ""; #create a string to store most popular name



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

        



for name in boys.keys():

    if name in girls.keys(): #Check to see if names are present in both girl and boy names

        if popular == "": #store first name in popular variable to compare in future

            popular = name

        if girls[name] > girls[popular]: #replace popular variable if a more popular name comes up

            popular = name

            

#print the most popular name

print("The most popular girl name that is also a boy name is: " + popular)

            
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



#Searches name inputted by user and outputs how many babies have that name

def search_name():         

    print("What name are looking for?")

    lookup = input()

    girl_name = False; #track if name is present in girls name dictionary

    boy_name = False; #track if name is present in boys name dictionary



    for name in girls.keys():

        if lookup.lower() == name: #if there is a match, print the number of girls with name

            girl_name = True

            print("There are %d girls with the name " %girls[name])

    if girl_name == False: #print when there is no match

        print("There are no girls with that name.")

    

    #loop through boys dictionary

    for name in boys.keys():

        if lookup.lower() == name: #if there is a match, print the number of boys with name

            boy_name = True 

            print("There are %d boys with the name " %boys[name])

    if boy_name == False: #print when there is no match

        print("There are no boys with that name.")



search_name()
