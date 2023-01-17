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

#1. Are there more boy names or girl names? What about for particular first letters? What about for every first letter?



#Since the three variables in the dataset have already been defined in example 1, I can use those established variables as a way to plug in my calculations.

# I've set up a conditional that checks the difference between the total number of boys and girls. It assigns that difference to x and surfaces it in human readable string.

if len(girls) > len(boys):

    x = len(girls) - len(boys)

    print("There are", x, "more girls")

#I could have made the elif statement linked to x. Something like "if x is < 0 then print there are more boys". However, I decided to just create a new variable

#because I didn't want to have to deal with negative numbers.

elif len(girls) < len(boys):

    y= len(boys)-len(girls)

    print("There are", y, "more boys")



#What about for particular first letters

x=input() #Assigning the user input to a variable

boysFirstLetter = {} #creating a dictionary that will eventually ingest what names match the user input

girlsFirstLetter = {} #same for girls

for name in boys.keys(): #utilizing the variables already created in example 1 to scan each line of the boys dictionary

    if name[0] == x:    # If the first string in a boys name equals the input

        boysFirstLetter[name.lower()] = count # store the boys name in the dictionary, with its count

for name in girls.keys():

    if name[0] == x:    # If the first string in a boys name equals the input

        girlsFirstLetter[name.lower()] = count # store the boys name in the dictionary, with its count

print("There are", len(boysFirstLetter), "boys with", x, "as their first letter") #printing the output

print("There are", len(girlsFirstLetter), "girls with", x, "as their first letter")



#What about for every first letter?



        
#What about for every first letter?



boysFirst = [] #list I am going to put every boy name that starts with the first letter of each name for

girlsFirst = [] #list I am going to put every girl name that starts with the first letter of each name for

letter=("a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z") #inefficiently referencing every letter

start = 0 #variable I am planning on using to iterate through the 'letter' list



for name in boys.keys(): #utilizing the variables already created in example 1 to scan each line of the boys dictionary

    if name[0] == letter[start]: #going through every line of the of the boy dictionary and referencing it against the 0 index of the letter list

        boysFirst.append(name) #adding each name that was found to the boysFirst list. 

print("There are", len(boysFirst), "boys with", letter[start], "as the first letter") #printing the results



for name in girls.keys(): #utilizing the variables already created in example 1 to scan each line of the boys dictionary

    if name[0] == letter[start]: #going through every line of the of the boy dictionary and referencing it against the 0 index of the letter list

        girlsFirst.append(name) #adding each name that was found to the boysFirst list. 

print("There are", len(girlsFirst), "girls with", letter[start], "as the first letter") #printing the results



#I suspect I need another loop here that goes through the letter list and creates a new list for each variable. However, I'm unsure how to do that right now. 





#What is the longest name in the dataset?

longNameBoy= [] #creating lists to drop the length of each name into

longNameGirl= []



for name in boys.keys(): #scanning each line of the dictionary

    x = len(name) #assigning the variable x with the length of each name

    longNameBoy.append(x) #dropping the length of the name into a list

    if x == 15: #since I know that the longest name has 15 characters, I'm going to set my list up to listen for this number. It's not very flexible, but it works.

        print(name) #Print the name that is associated with length of 15 characters.

print("These names above all have", max(longNameBoy),"letters. This is the most in the boy list") #this is where I cheated by finding the length of the longest name, I then listened for that length in a for loop



for name in girls.keys():

    y = len(name)

    longNameGirl.append(y)

    if y == 14:

        print(name)

print("These names above all have", max(longNameGirl),"letters. This is the most in the girl list") #this is where I cheated by finding the length of the longest name, I then listened for that length in a for loop

#5. How many names are subsets of other names?

x=0



for name in boys.keys():

    if name in boys.keys():

        x=x+1

    if name in girls.keys():

        y=y+1

print(x+y, "names are in other names")