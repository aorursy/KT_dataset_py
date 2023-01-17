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

# 2. How many babies are in the dataset (assuming nobody is counted more than once)?



# First taking the value of each of the boys' names and adding them together to create a grand total of boys' names.

# I do this by creating a variable for the total number of boys, set it to zero, then make a for loop going through 

# each value in the dictionary and adding it to the previous value until there are no more keys left.

boys_total = 0

for name in boys.keys():

    boys_total = int(boys[name]) + boys_total



#Now I do the same for girls, going through each key and value pair and making a total of girl babies.

girls_total = 0

for name in girls.keys():

    girls_total = int(girls[name]) + girls_total



# As I print out the answer, I add the totals together.

print("There are " + str(boys_total + girls_total) + " babies in the dataset.")

#3. What is the longest name in the dataset?



#Below you can see that I took your first example of making two dictionaries out of the dataset--one for boys and

#the other for girls-- and used part of it to make one list of all the names, ignoring the

#gender and count. I did this by appending the names to a new list called all_names.



NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



all_names = []  # create an empty list of keys for boys and girls to be able to compare length





for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")

    

    all_names.append(name)



#Now that I have a list of all the names, I need to keep track of the length of each, the name associated with the

#length, and I need to make a new list of just the longest names, in case there is more than one.  These three

#variables are seen below.



name_length = len(name)

long_name = name

long_names = []



#Now I make a For loop to go through all the names in the list, finding the first one that is the longest and

#assigning it to the long name variable. Now I know that there are no names longer than this one, but there

#still could be a name just as long later in the list. So I do another for loop and for every name that is 

#equally long, I add to the list I made. Lastly I print the list.



for name in all_names:

    if len(name) > name_length:

        name_length = len(name)

        long_name = name



for name in all_names:

    if len(name) == len(long_name):

        long_names.append(name)

        

        

print("The longest name or names in the dataset are " + str(long_names))  
#7. Write a program that will take a name as input and return the number of babies with that name in the girl and boy datasets.



#Here I started with the boy/girl dictionaries that were made earlier.  After using the input function to make user input

#equal to its variable, I also created a boolean for keeping track of if the input is in the dictionaries at all. 



input_name = input("Enter a name: ")

name_in_dict = False



#Then, I used a for loop to go through each dictionary and if I found the input name, I would print how many babies with that

#name were found in that dictionary.  I made sure to lower case the input name to match the all lower case names in the

#dictionaries. Last, if the input name isn't found in any of the dictionaries, I made that variable equal to false and 

#printed that there are no babies with the name.





for name in boys.keys():

    if name == input_name.lower():

        print("There are " + str(boys[name]) + " boys with that name.")

        name_in_dict = True

        

for name in girls.keys():

    if name == input_name.lower():

        print("There are " + str(girls[name]) + " girls with that name.")

        name_in_dict = True



if name_in_dict == False:

    print("There are no babies with that name.")

    