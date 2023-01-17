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

#state a variable "namesofboys" to be used for values of this list

namesofboys = boys.values()

#state a variable "namesofgirls" to be used for values of this list

namesofgirls = girls.values()



#set this variable to combine the length of the boys and girls list

totalnumberofnames = sum(namesofboys) + sum(namesofgirls)



#print a proper sentence with spacing between the phrases and symbols and include what was defined above

print("Number of babies in the dataset" + " " + "=" + " " + str(totalnumberofnames))
#create a for loop for the boys dataset because i couldn't figure out how to mimic it for girls for some reason

for name in boys.keys():

    #state a variable to equal to the length of any name with an index that extends beyond 0

    lengthofanyname = (len(name[0:]))

    #i just increased this number until the list stopped showing names, the list stops showing names at 15

    lengthoflongestname = 14

    

    #constrain results based on where the length of a name is greater than the length of any name

    #new variable introduced to define the longest name

    if lengthofanyname > lengthoflongestname:

        #constrain results based on what also equals the same length of these two variables

        #the results must meet both of these conditions, so it would sort based on where they equal, and where one is longer, in a loop

        lengthofanyname = lengthoflongestname

        #print the names that meet both of these conditions, as that should be the longest name

        print(name)

        #couldn't figure out how to make the list of names a proper list in a sentence so i rephrased the sentence

        print("One of the longest names in the dataset is", name + ".")
#state a variable to refer to all the keys in the boys dictionary

boysnames = set(boys.keys())

#state a variable to refer to all the keys in the girls dictionary

girlsnames = set(girls.keys())



#use the intersection function within python to find where they overlap

overlap = boysnames.intersection(girlsnames)



#print the number of names that overlap and complete a sentence to state the answer

print(len(overlap), "names overlap with both girls and boys.")
#state a variable that asks for input with the question of entering some baby name

somespecificname = input("Enter a name that a baby may have been named in 2010:")

#set all results to lowercase

enteredname = somespecificname.lower()

#state a variable for the keys in the boys list

for anyname in boys.keys():

    #constrain by what is entered and what names exist in the keys

    if enteredname == anyname:

        #the total number of babies with this name are calculated by referring to the key that matches the entered name

        numberofboyswiththisname = boys[anyname]

#state a variable for the keys in the boys list        

for anyname in girls.keys():

    #constrain by what is entered and what names exist in the keys

    if enteredname == anyname:

        #the total number of babies with this name are calculated by referring to the key that matches the entered name

        numberofgirlsswiththisname = girls[anyname]

        

    #if the input is not found, print this error message

    else:

        print("Nope, sorry, that baby name wasn't used in 2010.")

        #break keeps this from repeating itself

        break

        

#add the total of results to create a third variable that represents the total number of names

numberofbabieswiththisname = numberofboyswiththisname + numberofgirlsswiththisname       

#write a proper sentence of what the total is 

print("There are a total of", numberofbabieswiththisname, "babies with this name from 2010.")