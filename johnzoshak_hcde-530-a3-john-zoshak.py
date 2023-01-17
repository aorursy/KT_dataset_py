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

#Exercise 7

def baby_finder(x):

    results = [] # creates an empty array for holding values

    x = x.lower() # lower-cases input so people can type ArYa

    for name in boys.keys(): # iterates through the keys in boys looking for an exact match of the name

        if x == name:

            results.append(str(boys[name])) # appends to the end of my results string the value of the matching key.

    

    if len(results) != 1: # hacky check to see if a value was added, if not it adds a 0 to accurately report there were no matches.

        results.append("0")

            # I'm sure there's a better way to write this, I'll think about it some more

    for name in girls.keys(): #same as above but for the girls.

        if x == name:

            results.append(str(girls[name]))

    

    if len(results) != 2: # same hacky check to make sure something was appended.

        results.append("0")

    

    return "There are " + results[0] + " boys and " + results[1] + " girls named " + x + " in the data set" # prints results. 



print(baby_finder("Arya"))

    

    
#exercise 2

boy_count = 0 #sets my variables

girl_count = 0 



for name in boys.keys(): #iterates through the boy keys and adds the count of each successive loop to the last one.

    boy_count = boy_count + boys[name]



for name in girls.keys(): #same as above except with the girl keys

    girl_count = girl_count + girls[name]



total_count = girl_count + boy_count #adds the two variables together to get the final count. 



print("There are " + str(total_count) + " babies in the dataset") #prints out my results with some flourish. 

    
longest_name = "" # creates empty string



for name in boys.keys(): #iterates through the boys dictionary and replacing the variable with the name if its length is longer than the last one checked

    if (len(longest_name) < len(name)):

        longest_name = name



for name in girls.keys(): # same but with the girl data. 

    if (len(longest_name) < len(name)):

        longest_name = name



print(longest_name) #prints the longest name