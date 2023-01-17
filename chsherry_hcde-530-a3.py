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

#2. How many babies are in the dataset (assuming nobody is counted more than once)?
fname = "/kaggle/input/a3data/yob2010.txt"
count = 0
with open(fname, 'r') as f:
    for line in f:
        count += 1
print("Total of baby names is", count)

#using a loop to count the number of lines, assuming that each line is a baby name


#4. How many boy names are also girl names? How many girls' names are also boys' names?

count= 0 
for name in boys.keys():
    if name in girls.keys():
        count +=1
print (count)
 

count= 0 
for name in girls.keys():
    if name in boys.keys():
        count +=1
print (count)
        
#set count to count the number of results returned for comparing dictionaries of boys and girls names

   

#3.What is the longest name in the dataset?

longest_name=""
for line in open(NAMES_LIST, 'r').readlines(): 
    name, gender, count = line.strip().split(",")
    if len(longest_name) < len(name):
         longest_name=name
print(longest_name)
    


#set longest name to a null value to compare it to each name in the dataset
#iterate through the lines and separate each strings into three variables (name, gener count)
#using for loop to continue to compare until the longest_name is greater than name, and then it prints longest name

