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
for name in boys.keys():# This could be removed!!
    if name == "john":# This could be removed!!
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

# We create a nested dictionary that containst 2 keys, one for boys, one for girls.
# Each corresponds to another dictionary that contain the count of each initial for boy and girl names.
# https://www.geeksforgeeks.org/python-nested-dictionary/
iCount = {'boys':{}, 'girls':{}}

# Iterate through all the boy names and +1 to the initial count corresponding to the initial of the name
for name in boys:
    # If we already have a key for it, just add 1, otherwise, initialize it as 1.
    if name[0] in iCount['boys']:
        iCount['boys'][name[0]] += 1
    else:
        iCount['boys'][name[0]] = 1

# Iterate through all the girl names and +1 to the initial count corresponding to the initial of the name
for name in girls.keys():
    # If we already have a key for it, just add 1, otherwise, initialize it as 1.
    if str(name[0]) in iCount['girls']:
        iCount['girls'][name[0]] += 1
    else:
        iCount['girls'][name[0]] = 1

# We will store the total count in the next variables to compare it with the "Example 1 counts" and see if our
# answers look correct.
countB = 0 # Total count boys
countG = 0 # Total count girls

##### print('Boy Initial Count:')
# Iterate over a sorted list of boys initals and print their counts
for i in sorted(iCount['boys']):
    ##### print(i , " : " , iCount['boys'][i])
    countB+=iCount['boys'][i]
# Confirmation
##### print("Boy counts match?", countB == len(boys))

##### print('\nGirls Initial Count:')
# Iterate over a sorted list of boys initals and print their counts
for i in sorted(iCount['girls']):
    ##### print(i , " : " , iCount['girls'][i])
    countG+=iCount['girls'][i]
# Count confirmation
##### print("Girl count match?", countG == len(girls))
##### print("\n")

# Fuction that returns the gender that has more names with the provided initial
def biggerGender(initial):
    # If more boys, print "Boy", if more girls, print "Girl", if none is bigger, then print "Both"
    if iCount['boys'][initial] > iCount['girls'][initial] :
        return 'Boy'
    elif iCount['boys'][initial] < iCount['girls'][initial]:
        return 'Girl'
    else:
        return 'Both'
        
#######################################
# Time to answer the questions
#######################################
print("***\tQuestions/Answers\t***\n\n")

##########
# Are there more boy names or girl names?
##########
print("-Are there more boy names or girl names?")

# If the lenght of boy names is bigger than the lenght of girl names, print that there are more boy names, otherwise,
# check if the length of the girl names is bigger than the boys names. If that is also false, then it means they are equal.
if len(boys)>len(girls):
    print("  There are more boy names than girl names.\n")
elif len(boys)<len(girls):
    print("  There are more girl names than boy names.\n")
else:
    print("  There are the same amount girl names than boy names.\n")

##########
# What about for particular first letters?
##########
print("-What about for particular first letters?")

# Print the results for 'i', 'd', 'z', 'r', and 'u'
initial = 'i'
print("  The gender with more names with the initial '{}' is:\n    {}\n".format(initial, biggerGender(initial)))
initial = 'd'
print("  The gender with more names with the initial '{}' is:\n    {}\n".format(initial, biggerGender(initial)))
initial = 'z'
print("  The gender with more names with the initial '{}' is:\n    {}\n".format(initial, biggerGender(initial)))
initial = 'r'
print("  The gender with more names with the initial '{}' is:\n    {}\n".format(initial, biggerGender(initial)))
initial = 'u'
print("  The gender with more names with the initial '{}' is:\n    {}\n".format(initial, biggerGender(initial)))

##########
# What about for every first letter?
##########
print("-What about for particular first letters?")

# Dictionary containg the gender with more names for each initial
biggerGenderIntials = {} 
# We know (due to the confirmations above) that both boys and girls contain names with each letter of the alphabet
# for this reason we can just iterate through 1 of the intial list and compare with the other one, knowing that we
# won't be missing any letters.
for i in iCount['girls']:
    biggerGenderIntials[i] = biggerGender(i)

# Iterate thouth the sorted keys of the initial comparison result(biggerGenderIntials) and print their key/value pairs.
for i in sorted(biggerGenderIntials):
    print("  ", i , " : " , biggerGenderIntials[i])    
# How many babies are in the dataset (assuming nobody is counted more than once)?

# We define or babies total variable.
babiesTotal = 0

# Iterate through the dictionaries of boys and girls and add the counts to the total babies.
for name in boys:
    babiesTotal += boys[name]

for name in girls:
    babiesTotal += girls[name]

#######################################
# Time to answer the questions
#######################################
print("***\tQuestions/Answers\t***\n\n")

##########
# How many babies are in the dataset?
##########
print("-How many babies are in the dataset?")
print("  ", str(babiesTotal))
    
# My longestName variable will be an list because there may be more than one longest name.
# I'll initialized my list with an empty string '' so that I have something to compare to.
longest_names = ['']

# Iterate through each boy name to compare them to the Longest Name so far
for name in boys:
    # If the name is longer than the Longest Name so far, replace the longest_names array with an array containing this new longer name.
    # otherwise, if the name is the same length, append it to the array.
    if len(name)>len(longest_names[0]):
        longest_names = [name]
    elif len(name)==len(longest_names[0]):
        longest_names.append(name)

# Iterate through each girl name to compare them to the Longest Name so far
for name in girls:
    # If the name is longer than the Longest Name so far, replace the longest_names array with an array containing this new longer name.
    # otherwise, if the name is the same length, append it to the array.
    if len(name)>len(longest_names[0]):
        longest_names = [name]
    elif len(name)==len(longest_names[0]):
        longest_names.append(name)

#######################################
# Time to answer the questions
#######################################
print("***\tQuestions/Answers\t***\n\n")

##########
# What is(are) the longest name(s) in the dataset?
##########
print("-What is(are) the longest name(s) in the dataset?")
print('  ', longest_names)
# The total count of gender neutral names.
gender_neutral_count = 0

# Iterate through the boy names and check if the name also exist in the girl names.
# if it does, add 1 to the Gender Neutral Count
for name in boys:
    if name in girls:
        gender_neutral_count+=1

#######################################
# Time to answer the questions
#######################################
print("***\tQuestions/Answers\t***\n\n")

##########
# How many boy names are also girl names? How many girls' names are also boys' names?
##########
print("-How many boy names are also girl names? How many girls' names are also boys' names?")
print('  {} boy names are girl names and viseversa'.format(gender_neutral_count))
# The total count of subset names
subset_names = 0
# A combined list of names(both boys and girls), fromkeys() is helping me remove duplicates.
bgs = list(dict.fromkeys(list(boys) + list(girls)))

# OLD Solution
# Iterate through the combined list of names then iterate through the list again with a nested for loop and check the "outer name" against the "inner name"
# to determine if it is a subset.
# for name1 in boysandgirls:
#    for name2 in boysandgirls:
#        # If it is a subset, then check if the name is not the same and add 1 to the total Subset Names count if both things are true.
#        if name1 in name2 and name1 != name2:
#            ##### print(name1 + " - " + name2)
#            subset_names+=1
#            break;

# Faster Solution
# Loop until we run out of elements, i.e. our list length is down to 1. This last element doesn't 
# We will use an index 0 and a changing "i" to check for subsets and we will be poping(removing) the elements that meet certain conditions.
i=1 # Our comparison index
while len(bgs) != 1:
    if bgs[0] in bgs[i]:# If 0 is subset of element "i", we remove element 0.
        ##### print(bgs[0] + " - " + bgs[i])
        bgs.pop(0)
        # Add 1 to the subset total count and restart the "i" to restart the search because element "0" has changed.
        subset_names+=1
        i=1
    elif bgs[i] in bgs[0]:# if element "i" is subset of element 0, we remove element "i".
        ##### print(bgs[i] + " - " + bgs[0])
        bgs.pop(i)
        # Add 1 to the subset total count, but do not restart the search(no reset to the "i"), we still need to keep looking to know if element "0" is a subset.
        subset_names+=1 
    elif i<len(bgs)-1: # If my "i" is not out of range yet, increase my index and keep comparing.
        i+=1
    else: # Otherwise remove the then element 0 and restart "i" to reset the comparisons
        bgs.pop(0)
        i=1

#######################################
# Time to answer the questions
#######################################
print("***\tQuestions/Answers\t***\n\n")

##########
# How many names are subsets of other names?
##########
print("-How many names are subsets of other names?")
print("  ",str(subset_names)) # Result = 10090
##### print(girls['ana'])
# Our most popular name. I am initializing it with my name which I confirm it exists with the print above. If it's not the most popular, it will get replaced.
most_popular = ['ana']
# Iterate through the boy names and check if the name also exist in the girl names.
for name in girls:
    if name in boys:
        # Now check if the name popularity is greater than the most popular so far.
        # If yes, replace it, if is the same, append it to the Most Popular list, otherwise, ignore it.
        if girls[name] > girls[most_popular[0]]:
            most_popular = [name]
        elif girls[name] == girls[most_popular[0]]:
            most_popular.append(name)

#######################################
# Time to answer the questions
#######################################
print("***\tQuestions/Answers\t***\n\n")

##########
# What is the most popular girl name that is also a boy name?
##########
print("-What is(are) the most popular girl name(s) that is also a boy name?")
print("  ",most_popular)
# Function that returns a dictionary with the number of boy names and the number of girl names.
# I used the get() function istead of the normal [] index reference because I want to be able to set a default value(0) and avoid hitting a key error.
def babies_with_name(name):
    return {'boys': boys.get(name,0), 'girls': girls.get(name,0)}

#######################################
# Time to answer the questions
#######################################
print("***\tQuestions/Answers\t***\n\n")

##########
# What is the most popular girl name that is also a boy name?
##########
print("-Write a program that will take a name as input and return the number of babies with that name in the girl and boy datasets.")

# Print the results for 'isabella', 'john', 'ana', 'ali', 'brock', 'federico', and 'amelia'
name = 'isabella'
print("  There were\n    " + str(babies_with_name(name)['boys']) + " boys and " + str(babies_with_name(name)['girls']) + " girls\n\tnamed '" + name +"'.")
name = 'john'
print("  There were\n    " + str(babies_with_name(name)['boys']) + " boys and " + str(babies_with_name(name)['girls']) + " girls\n\tnamed '" + name +"'.")
name = 'ana'
print("  There were\n    " + str(babies_with_name(name)['boys']) + " boys and " + str(babies_with_name(name)['girls']) + " girls\n\tnamed '" + name +"'.")
name = 'ali'
print("  There were\n    " + str(babies_with_name(name)['boys']) + " boys and " + str(babies_with_name(name)['girls']) + " girls\n\tnamed '" + name +"'.")
name = 'brock'
print("  There were\n    " + str(babies_with_name(name)['boys']) + " boys and " + str(babies_with_name(name)['girls']) + " girls\n\tnamed '" + name +"'.")
name = 'federico'
print("  There were\n    " + str(babies_with_name(name)['boys']) + " boys and " + str(babies_with_name(name)['girls']) + " girls\n\tnamed '" + name +"'.")
name = 'amelia'
print("  There were\n    " + str(babies_with_name(name)['boys']) + " boys and " + str(babies_with_name(name)['girls']) + " girls\n\tnamed '" + name +"'.")