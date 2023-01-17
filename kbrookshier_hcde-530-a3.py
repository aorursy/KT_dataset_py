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

# Exercise 1



# Are there more boy names or girl names?

# Print number of unique boy and girl names

print("There are " + str(len(boys.keys())) + " unique boy names in the dataset.")

print("There are " + str(len(girls.keys())) + " unique boy names in the dataset.")



# Specify which has the greater number of unique names

if len(boys.keys()) > len(girls.keys()):

    print("There are more unique boy names")

elif len(girls.keys()) > len(boys.keys()):

    print("There are more unique girl names")

else:

    print("There are an equal number of boy and girl names")

    

print()





# What about for particular first letters? What about for every first letter?

# Create dictionaries for letters

boys_first_letter = {}

girls_first_letter = {}



# Function builds dictionary with keys

def build_first_letter_dict(gender):

    # Create a dictionary of letters

    first_letter = {}

    

    # Loop through all names in the txt file

    for name in gender.keys():

        letter = name[0]               # get the first letter of the name



        if letter in first_letter:

            first_letter[letter] += 1  # if the letter exists in the dict, add 1 to the count

        else:

            first_letter[letter] = 1   # if the letter does not exist, add it to the dict

            

    return(first_letter)



# Create boy and girl first letter dicts

boy_first_letters = build_first_letter_dict(boys)

girl_first_letters = build_first_letter_dict(girls)



# For each letter of the alphabet, specify if there are more unique girl or boy names w/ that first letter

for letter in 'abcdefghijklmnopqrstuvwxyz':

    boy_count = boy_first_letters[letter]

    girl_count = girl_first_letters[letter]

    

    if boy_count > girl_count:

        print("More unique boy names start with", letter.upper(), ';', boy_count, 'M vs.', girl_count, 'F')

    elif girl_count > boy_count:

        print("More unique girl names start with", letter.upper(), ';', boy_count, 'M vs.', girl_count, 'F')

    else:

        print("Equal number of unique boy and girl names start with", letter.upper())

        
# Exercise 2

# How many babies are in the dataset (assuming nobody is counted more than once)?



# Start count for each gender at zero

count_boys = 0

count_girls = 0



# Iterate through the dataset

for line in open(NAMES_LIST, 'r').readlines():  

    name, gender, count = line.strip().split(",")  # Split each line into its data components

    

    if gender == 'F':

        count_girls += int(count)

    if gender == 'M':

        count_boys += int(count)



# Print the results of the counts

print("There were", count_girls, "girls with names that occurred 5+ times nationally in this dataset.")

print("There were", count_boys, "boys with names that occurred 5+ times nationally in this dataset.")
# Exercise 3

# What is the longest name in the dataset?



longest_name_length = 0

longest_name = ''



for line in open(NAMES_LIST, 'r').readlines():  

    name, gender, count = line.strip().split(",")  # Split each line into its data components

    

    if len(name) > longest_name_length:            # If the current name is longer than the longest name

        longest_name = name                        # Then, set the current name to the longest name

        longest_name_length = len(name)            # Then, set the current name length to longest name length

    else:

        pass                                       # Otherwise, move on



print(longest_name, longest_name_length)
# Exercise 4

# How many boy names are also girl names? How many girls' names are also boys' names?



girls_names_that_are_boys_names = []    # Create an empty list of names



for name in girls.keys():               # For each name in the girls' dict

    if name in boys.keys():             # Check if that name exists in the boys' dict

        girls_names_that_are_boys_names.append(name)    # If both exist, add the name to the list



# Print how many names are both boy and girl names

# By virtue of names being on both lists, they will have the same length of boy/girl co-name lists

len(girls_names_that_are_boys_names)