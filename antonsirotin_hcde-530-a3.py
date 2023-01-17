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

# checks if there are more names in dict.list[boys] and prints statement if true
if len(boys.keys()) > len(girls.keys()):
    print("More boy names than girl names")

# checks if there are more names in dict.list[girls] and prints statement if true
elif len(girls.keys()) > len(boys.keys()):
    print("More girl names than boy names")
    
# checks if there are the same # of names in both lists and prints statement if true
elif len(boys.keys()) == len(girls.keys()):
    print('Same number of boy and girl names')

#declare counts for boy and girl names starting with letter A
boysCount = 0
girlsCount = 0

#declare variable with alphabet through which to iterate
alph = 'abcdefghijklmnopqrstuvwxyz'


#scan through alphabet
for letter in alph:    
#scan through a list of all boy names and look at first letter. If it's a particular letter, add to count and print count
    for nameB in boys.keys():
        if nameB[0].lower() == letter:
            boysCount += 1
    print('boys names starting with', letter, '-', boysCount)
    boysCount = 0
    

#scan through a list of all girl names and look at first letter. If it's a particular letter, add to count and print count
    for nameG in girls.keys():
        if nameG[0].lower() == letter:
            girlsCount += 1
    print('girls names starting with', letter, '-', girlsCount)
    girlsCount = 0

#compare each count var. If there are more girl or boy 'A' names, or if they are equal, print corresponding statement
if boysA > girlsA:
    print("More boys names starting with 'A'")
elif girlsA > boysA:
    print("More girls names starting with 'A'")
elif girlsA == boysA:
    print("Same number of girls and boys names starting with 'A'")

#declare var for count
total = 0

#iterate through every boy name and add the value of each key to total
for name in boys:
    total += boys[name]
    
#iterate through every girl name and add the value of each key to total
for name in girls:
    total += girls[name]
    
#prints total number of babies
print('There are', total, 'babies')
#declare input variable with 
inputName = input('Please enter a name.\n')

#iterates through boys dictio
for name in boys:
    if name == inputName:
        print('there are', boys[inputName], 'boys with that name')
for name in girls:
    if name == inputName:
        print('there are', girls[inputName], 'girls with that name')

print('there are', boys[inputName] + girls[inputName], 'babies with the name', inputName)
    