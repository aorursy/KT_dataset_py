

        

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

if len(girls) < len(boys): #counts whether there are more boy names or girl names

    print ("There are more boy names than girl names.")

else:

    print ("There are more girl names than boy names.")
a_girlnames = 0 

for name in girls.keys():

    if name[0] == 'a': # if the first letter in the name is an 'a'

        a_girlnames += girls[name] # add one to total number of girl names that start with an a

print("There is a total of " + str(a_girlnames) + " girl names that start with an a.")



a_boynames = 0

for name in boys.keys():

    if name[0] == 'a': # if the first letter in the name is an 'a'

        a_boynames += boys[name] # add one to total number of girl names that start with an a

print("There is a total of " + str(a_boynames) + " boy names that start with an a.")
def girls_list():

    girlnames = []

    for key in girls.keys():

        girlnames.append[key]

    return girlnames

    

girls_list() # trying to take just the names from girls' dictionary and turn it into a list, but is not working here
g_total = 0

b_total = 0



for name in girls.keys():

    g_total += girls[name]

print ("There is a total of " + str(g_total) + " girls.")



for name in boys.keys():

    b_total += boys[name]

print ("There is a total of " + str(b_total) + " boys.")



print ("This is the total number of babies: "+ str(g_total + b_total))
best = 0

for name in girls.keys():

    if len(name) > best: # if the length of name is longer than whatever the length of the longest name is so far

        best = len(name) # then update the value of 'best' and set it as the new longest length 

        longest_girl = name # get the longest name 

print (longest_girl, best)



for name in boys.keys(): # doing the same as above for boy names 

    if len(name) > best:

        best = len(name)

        longest_boy = name

print (longest_boy, best)



print ("The longest name is: " + str(longest_boy))
same = 0

for name in boys.keys(): # for each name in boys' names

    if name in girls.keys(): # if that name is also in girls' list of names

        same += 1 # then add one

print("This is the number of boys' names that are also in girls' names: " + str(same)) 



same2 = 0

for name in girls.keys(): # for each name in girls' names

    if name in boys.keys(): # if that name is also in boys' list of names

        same2 += 1 # then add one

print("This is the number of girls' names that are also in boys' names: " + str(same2))
subset = 0

for name in boys.keys(): # for each name in boys' list of names

    if name in boys.keys(): # check if that name is a part of other names

        subset += 1

print("There are " + str(subset) + " names that are also a part of other names.")
popular_count = 0

for name in girls: # for each girl name

    if name in boys: # look for if that girl name is also in list of boy names

        values = girls[name] # if yes, then get the values of overlapping names

        if values > popular_count:

            popular_count = values

            popular_name = name

print("This is the most popular girl AND boy name, with the count: " + str(popular_name) + ", " + str(popular_count))