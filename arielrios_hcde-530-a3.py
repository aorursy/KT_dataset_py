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

boyscount = len(boys)

girlscount = len(girls)



#print ("the boys list has " + str(boyscount))

#print ("the girls list has " + str(girlscount))

#print()



if boyscount == girlscount:

    print ("There are the same number of names for both boys and girls.")

elif boyscount > girlscount:

    print ("There are more boys names than girls.")

else:

    print ("There are more girls names than boys names.")

totalnames = boyscount + girlscount

print ("There are " + str(totalnames) + " names.")
namefind = "ben"

namefindcounter = 0



for name in boys.keys():

    if namefind in name:

        namefindcounter += 1

        #print(name + " " + str(boys[name]))



for name in girls.keys():

    if namefind in name:

        namefindcounter += 1

        #print(name + " " + str(girls[name]))

print ("The name " + namefind +" appears " + str(namefindcounter) + " times in the baby list's." )
namefind = "ben" # Don't know how to get an input in Kaggle

namefindcounter = 0



for name in boys.keys():

    if name.startswith(namefind):

        namefindcounter += 1

        # print(name + " " + str(boys[name]))



for name in girls.keys():

    if name.startswith(namefind):

        namefindcounter += 1

        #print(name + " " + str(girls[name]))

print ("There are "  + str(namefindcounter) + " names that begin with " + namefind +" in the baby list's." )