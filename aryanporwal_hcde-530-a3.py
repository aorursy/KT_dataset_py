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

#defined a variable to store the total number of babies
total_babies = 0
# iterate through the boys dictionary. for each key add the values to the total_babies variable
for name in boys.keys():
    total_babies += boys[name]

# iterate through the girls dictionary. for each key add the values to the total_babies variable
for name in girls.keys():
    total_babies += girls[name]

# print the total number of babies    
print("Number of babies in the dataset are", total_babies)
#defined a variable to store the number of characters in a name
max_char = 0
# iterate through the boys dictionary. for each key check the character count
for name in boys.keys():
    # if the length is greater than the stored value in the variable, replace the variable value with the max new value
    if len(name) > max_char:
        max_char = len(name)
        maxword = name # Use a variable to also store the name value

for name in girls.keys():
    # if the length is greater than the stored value in the variable, replace the variable value with the max new value
    if len(name) > max_char:
        max_char = len(name)
        maxword = name # Use a variable to store the name value
# Print the longest name
print(f"{maxword} is the longest name with {max_char} characters.")
#defined a variable to store the number of boy names in girl names dictionary
count = 0
# iterate through the boys dictionary. for each key check if the name is also present in girls dictionary
for name in boys.keys():
    if name in girls.keys():
        # if the name is present increase the counter by 1
        count += 1
# Print the number of boy names that are also girl names
print(f"{count} boy names are also girl names.")

total_count=0
# iterate through the girls dictionary. for each key check if the name is also present in boys dictionary
for name in girls.keys():
    if name in boys.keys():
        #count the number of babies with girls' names that are also boys' names 
        total_count += girls[name]  
# Print the number of girls' names that are also boys' names
print(f"{total_count} girls' names are also boys' names.")
#defined a variable to store the count of poplar names in babies
popular_count = 0
# iterate through the girls dictionary. for each key check if the name is also present in boys dictionary
for name in girls.keys():
    if name in boys.keys():
        # Check the most popular girl name by comparing the value
        if girls[name] > popular_count:
            popular_count = girls[name]
            popular_name = name

# Print the most popular girl name that is also a boy name
print(f"{popular_name} is the most popular girl name that is also a boy name.")
#defined a variable to store the count of babies with the name same as imput name
number = 0
#ask a name to be entered
input_name = input ("Please input a name: ")
# iterate through the boys dictionary. for each key check if the name is same as the input name
for name in boys.keys():
    if name == input_name:
        # if the names are same, add the number of babies to the variable
        number += boys[name]  

# iterate through the girls dictionary. for each key check if the name is same as the input name        
for name in girls.keys():
     # if the names are same, add the number of babies to the variable
    if name == input_name:
        number += girls[name]    
        
# Check if baby with the same name is found. Print the number of babies else print babies with the name not found        
if number > 0:
    print(f"Number of babies with the name {input_name}: {number}")
else:
    print(f"No babies with the name {input_name} found")