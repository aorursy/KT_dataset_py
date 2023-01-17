NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"





boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls

all_names = {} #creating an empty dictionary of key:value pairs for all names





for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    # Since 'count' is actaully a string of text and not an integer, 

    # we need to turn it into an integer to store that number in the dictionary so we can use it. 

    # later to do arithmetic that we couldn't do, if it was just text. This is called 'casting'.

    

    count = int(count)   # Cast the string 'count' to an integer

    all_names[name.lower()] = count

    

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

print("There were %d total unique names." %len(all_names))



# We did this for you at the end of the previous homework.

# It's a little weird to stuff numbers into sentences this way, but once you get 

# used to it, it's easy. You can do lots of other formatting like this.

# Here's an explanation of how it works: https://www.geeksforgeeks.org/python-output-formatting/
# iterate through the boys dictionary. for each key see if it is: 'john'

def searchBoys():

    inputName = input("search boy names: ")

    for name in boys.keys():

        if name == inputName:

            # if it is 'john', get the value associated with john and use that value for the print statement

            # because the value is an integer, we have to cast it to a string in the print statement, with str().

            print("There were " + str(boys[name]) + " boys named " + name)



searchBoys()
for name in boys.keys():

    if name in girls.keys():

        print(name)
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))

# 1. Are there more boy names or girl names? What about for particular first letters? What about for every first letter?

alphabet = []

alphabet = [chr(x) for x in range(ord('a'), ord('z') + 1)] # found this elsewhere, didn't realize you could instantiate a list via a for loop within the list, which is cool



def greater_by_gender():

    if (len(boys) < len(girls)):

        print("There are more girl names than boy names.")

    elif (len(boys) > len(girls)):

        print("There are more boy names than girl names.")

    else:

        print("There are equal numbers of boy and girl names.")



def names_by_input_letter():

    search_letter = str(input("Search name list by letter"))

    counter = 0

    for name in all_names.keys():

        if name[0] == search_letter:

            counter += 1

    print("There are {} names that start with \'{}\'.".format(counter, search_letter))



def names_by_letters():

    for letter in alphabet:

        counter = 0

        for name in all_names.keys():

            if name[0] == letter:

                counter += 1

        print("There are {} names that start with \'{}\'.".format(counter, letter))





# Which gender has more names?

greater_by_gender()



# Choose your own letter adventure.

names_by_input_letter()



# Outputs string with all letters

names_by_letters()









# 2. How many babies are in the dataset (assuming nobody is counted more than once)?

boyCount = 0

girlCount = 0



for count in boys.keys():

    count = int(boys[count])

    boyCount += count



for count in girls.keys():

    count = int(girls[count])

    girlCount += count

    

total = boyCount + girlCount



print("total babies: {}".format(total))
# 3. What is the longest name in the dataset?

longest_names = []



def find_longest():

    longest_name_len = 0

    for name in all_names.keys():

        if (len(name) > longest_name_len):

            longest_name_len = len(name)



    for name in all_names.keys():

        if (len(name) == 15):

            longest_names.append(name)

            

    print(longest_names)

    

find_longest()

# 4. How many boy names are also girl names? How many girls' names are also boys' names?

overlap_count = 0

for name in boys.keys():

    if name in girls.keys():

        overlap_count += 1

        

print(overlap_count)
# 5. How many names are subsets of other names?



# this one is tripping me up. but I'd like to come back to it! Submitting at this point to make sure I get credit.