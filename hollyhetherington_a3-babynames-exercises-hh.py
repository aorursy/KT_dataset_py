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



boys = {} 

girls = {}



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
#Exercise 7

#I chose to find out how many instances there are of the name Ryan in the girls and boys dictionaries. 

#To do so, I will iterate through each dictionary looking for the name "Ryan" as a key 



for name in boys.keys():

    if name == "ryan":

        # if it is 'Ryan', get the value associated with that and use that value for the print statement

        # because the value is an integer, I will cast it to a string in the print statement, with str().

        print("There were " + str(boys[name]) + " boys named " + name)

#Now I will do the same for the girls dictionary, iterating through it for "ryan" as a key



for name in girls.keys():

    if name == "ryan":

        # if it is 'Ryan', get the value associated with that and use that value for the print statement

        # because the value is an integer, I will cast it to a string in the print statement, with str().

        print("There were " + str(girls[name]) + " girls named " + name)
#Exercise 6

# The answer is Jayden (there were 1,460 girls named Jayden, by far the most popular girls name that is also a boys name). Here's how I arrived at this. 



#First, I will try the code from the #3 example above (comparing dictionaries) to find which names appear in both dictionaries: 



for name in boys.keys():

    if name in girls.keys():

        print(name)
#Then, I will determine how many times each name appears in each dictionary. There's probably a faster way to do this...



# iterate through the boys dictionary. for each key see if it is: 'jacob'

for name in boys.keys():

    if name == "jacob":

        # if it is 'jacob', get the value associated with john and use that value for the print statement

        # because the value is an integer, we have to cast it to a string in the print statement, with str().

        print("There were " + str(boys[name]) + " boys named " + name)
# iterate through the girls dictionary. for each key see if it is: 'jacob'

for name in girls.keys():

    if name == "jacob":

        # if it is 'jacob', get the value associated with john and use that value for the print statement

        # because the value is an integer, we have to cast it to a string in the print statement, with str().

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "ethan":

        print("There were " + str(boys[name]) + " boys named " + name)



for name in girls.keys():

    if name == "ethan":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "michael":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "michael":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "jayden":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "jayden":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "william":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "william":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "alexander":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "alexander":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "noah":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "noah":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "daniel":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "daniel":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "aiden":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "aiden":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "anthony":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "anthony":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "joshua":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "joshua":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "mason":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "mason":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "christopher":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "christopher":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "andrew":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "andrew":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "david":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "david":

        print("There were " + str(girls[name]) + " girls named " + name)
for name in boys.keys():

    if name == "matthew":

        print("There were " + str(boys[name]) + " boys named " + name)

        

for name in girls.keys():

    if name == "matthew":

        print("There were " + str(girls[name]) + " girls named " + name)
#Exercise one and two#



#First, I tried to figure out how many total names are in girls dictionary and boys dictionary. 

#I'm going to use the code from example 1 because that results in a count from each name list. 

#However, it seems like not all of the code would be necessary to print the total from each list. 

#I tried simplifying by referencing the dictionary and running a count function.

#This is what I ran and it resulted in a count of the girls' names and boys' names. 

#Then I added the totals together. I'm sure there's a more sophisticated way to do this. 



for line in open(NAMES_LIST, 'r').readlines(): 

        name, gender, count = line.strip().split(",")



        count = int(count) 

    

if gender == "F":   

        girls[name.lower()] = count

elif gender == "M": 

        boys[name.lower()] = count



print("There were %d girls' names." %len(girls))

print("There were %d boys' names." %len(boys))



19791 + 14236



#Now I'm trying to figure out the first letter for every name in the girls dictionary. 

#I'm going to use sme of the code we learned in class for finding the first letter in a list or string. 

#I'm not really sure how to reference the lists/dictionaries in the code, though.

#I get a type error that says "list object is not callable"

#I googled the error and went to a huge Stackoverflow thread that wasn't really helpful/didn't make sense to me. 



#first attempt: 

print([0](NAMES_LIST))

#second attempt: 

print([0](girls))



#I don't understand why trying to figure out what you did wrong has to be so damned complicated. 