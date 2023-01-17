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



#use if/else if/else to compare the # of girl vs.boy names

if len(girls) > len(boys):

    print("There are more girl names")

elif len(boys) > len(girls):

    print("There are more boy names")

else:

    print("There is an equal amount of boy vs. girl names")

    

#Compare the amount of baby names starting with A for both genders

#create two lists that gathers the names starting with A for girls and boys

girlnameA = []

boynameA = []



for x in girls:

    if x[0] in "a":

        girlnameA.append(x)

for y in boys:

    if y[0] in "a":

        boynameA.append(x)



#Compare the length of the two list of names and print the approriate answer based on the result

if len(girlnameA) > len(boynameA):

    print ("There are more girls' name starting with 'A'")

elif len(girlnameA) < len(boynameA):

    print ("There are more boys' name starting with 'A'")

else:

    print("There is an equal amount of boys' and girls' name starting with 'A'")

    

print("...The last question confuses me. Are we suppose to go through all the alphabet?")
#create a list that includes all the values in girls

girlbabycount = []

for x in girls.values():

    girlbabycount.append(x)

#add up all the numbers in the list for girls

girltotal = 0

for xx in girlbabycount:

    girltotal = girltotal + xx



#now do the same process to calculate # of boy babies

boybabycount = []

for x in boys.values():

    boybabycount.append(x)

    

boytotal = 0

for xx in boybabycount:

    boytotal = boytotal + xx



#total the number of girl and boy babies

totalbabycount = girltotal + boytotal

print("There are", totalbabycount, "babies in the dataset.")

#compile lists for all the girl names with boy names

girlnames = []

for name in girls.keys():

    girlnames.append(name)

    

boynames = []

for name in boys.keys():

    boynames.append(name)



#combine the two list into one

totalnames = boynames + girlnames



#use sorted function to sort the list based on length from longest to shortest,

totalnames.sort(key=len, reverse = True)

#then run through the list of totalnames to see what names has the longest length,

#since the first name of totalnamesreverse defines the length of the longest name

for x in totalnames:

    if len(x) == len(totalnames[0]):

        print(x)
#compile lists for overlapping girl names with boy names

#not sure if i understood the question right here, cuz the two would have the same result

ginbname = []

for name in boys.keys():

    if name in girls.keys():

        ginbname.append(name)

print("There are", len(ginbname), "girl names that are also boy names.")



bingname = []

for name in girls.keys():

    if name in boys.keys():

        bingname.append(name)

print("There are", len(bingname), "boy names that are also girls names.")

# defines the input field

x = input("Enter name in lower case:")



#combine two dicts

girls.update(boys)



# use if/else to allow input be run through the dict

if x in girls.keys():

    print ("There are", girls[x], "babies name",x)

else:

    print("There are 0 babies name",x)