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

#create variable of longest length to use later.
longest_length = 0

# use for loop to iterate through the boy dictionary keys which are the boys names.
for name in boys.keys():
    
    #if the length of the name is greater than the previous, it becomes the new longest length.
    if len(name) > longest_length:
        longest_length = len(name)
        longest_boy_name = name

print('the longest boy name is ' + longest_boy_name)


#re-initialize longest_length variable
longest_length = 0

# use for loop to iterate through the girl dictionary keys which are the girls names.
for name in girls.keys():
    
    #if the length of the name is greater than the previous, it becomes the new longest length.
    if len(name) > longest_length:
        longest_length = len(name)
        longest_girl_name = name

print('the longest girl name is ' + longest_girl_name)


# if else statement to see which name is actually the longest between the longest boy and longest girl.
if len(longest_girl_name) > len(longest_boy_name):
    longest_overall_name = longest_girl_name
else:
    longest_overall_name = longest_boy_name
    
print('Therefore the longest overall name is ' + longest_overall_name)



        
        
#set a max count variable so we can access it later in the loop.
max_count = 0

# use a for loop to iterate through the girls dictionary keys.
for name in girls.keys():
    
    #if a girls name count is greater than the max count, it will become the new max count. This code also checks if the name exists in the boys dictionary.
    if girls[name] > max_count and name in boys.keys():
        max_count = girls[name]
        print(name)
        print(max_count)
    
    
#allow user to input name and set it as userinput
userinput = input("Please enter a name: ")

#account for uppercase use and convert everything to lower since that is how our dictionaries are set above.
inputname = userinput.lower()

#for loop to iterate through the boys dictionary keys.
for name in boys.keys():
    
    # if the input name matches the name that is being iterated on, then print out the name and it's count. Save the count for later.
    if inputname == name:
        print('There were ' + str(boys[name]) + ' boys named ' + inputname)
        boyCount = boys[name]
        
        #if the name is not in the dictionary key then print that there are no boys named the user's input.
        #break when this happens to prevent the loop to continue printing this on each iteration.
    elif inputname not in boys.keys():
        print('There are no boys named ' + inputname)
        break

        
##for loop to iterate through the girls dictionary keys.
for name in girls.keys():
    
    # if the input name matches the name that is being iterated on, then print out the name and it's count. Save the count for later.
    if inputname == name:
        print('There were ' + str(girls[name]) + ' girls named ' + inputname)
        girlCount = girls[name]
        
        #if the name is not in the dictionary key then print that there are no girls named the user's input.
        #break when this happens to prevent the loop to continue printing this on each iteration.
    elif inputname not in girls.keys():
        print('There are no girls named ' + inputname)
        break
        
        
totalnames = boyCount + girlCount
print('There were a total of ' + str(totalnames) + ' named ' + inputname)