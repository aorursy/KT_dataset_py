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

#Exercise 8: Take a prefix as input and print the number of babies with that prefix in each dataset (i.e., "m" would list babies whose names start with "m" and "ma" would list babies whose names start with "ma", etc).

#Variables to assign

prefix_numBoys=0

prefix_numGirls=0

input_prefix = "m"



#Where are the names that start with "m" in the first two positions in the boys dictionary? [0]

#Calculating the total 

for name in boys.keys():

    if input_prefix in name[0]:

        prefix_numBoys += 1   

    

#Where are the names that that start with "m" in the first two positions in the girls dictionary? [0]

#Calculating the total 

for name in girls.keys():

    if input_prefix in name[0]:

        prefix_numGirls += 1   

        

#Creating a variable for the total number of names that includes boys and girls with this "m" prefix.

total = (prefix_numBoys + prefix_numGirls)

   

print("Total number of babies in dataset with m as the prefix = " + str(total))



#New variable assignment to find a longer prefix and thus adjusting the totals   

prefix_ma_numBoys=0

prefix_ma_numGirls=0

input_prefix_ma = "ma"



#looking in the boys dictionary for names that start with "ma" in the first two position [0:2]

#running a total 

for name in boys.keys():

    if input_prefix_ma in name[0:2]:

        prefix_ma_numBoys += 1   

    

#Names that start with "ma" in the first two positions within the girls dictionary [0:2]

#Calculating the total

for name in girls.keys():

    if input_prefix_ma in name[0:2]:

        prefix_ma_numGirls += 1   

        

#assigning a variable to the total number that includes boys and girls with this prefix.

final_total = (prefix_ma_numBoys + prefix_ma_numGirls)

   

print("Total number of babies in dataset with ma as prefix = " + str(final_total))
#Excerise 4: How many boy names are also girl names? How many girls' names are also boys' names?

#First I will print all of the boys names that appear in the girls names dictionary and vice versa to verify the accuracy.

boys_names = 0

girls_names= 0



#Print all of the boys names that appear in the girls names list.



for name in boys.keys():

    if name in girls.keys():

        boys_names += 1

print(boys_names)

        

#Print all of the girls names that appear in the boys names list, to ensure they match. 



for nn in girls.keys():

    if nn in boys.keys():

        girls_names += 1

print(girls_names)
#Excercise 2: how many babies are in the dataset (assuming nobody is counted more than once)?

#For each boy's and girl's name in the dictionary, how many values are associated with each key?

boys_names = boys.values()



girls_names = girls.values()

   

#Summing the values of the boys and girls names in the dictionary  

print("Total number of babies in dataset= " + str(sum(girls_names)+sum(boys_names)))
