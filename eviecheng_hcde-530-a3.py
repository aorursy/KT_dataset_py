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

# Are there more boy names or girl names? 

if len(boys) > len(girls):

    print("There are more boy names than girl names.")

elif len(boys) < len(girls):

    print("There are more girl names than boy names.")

else:

    print("There are equal number of boy names and girl names.")

    

# What about for particular first letters? 

# I am trying with names with first letter 'a' here.

numboynames = 0

numgirlnames = 0

for name in boys.keys():

    if name.startswith('a'):

        numboynames = numboynames + 1

for name in girls.keys():        

    if name.startswith('a'):

        numgirlnames = numgirlnames + 1

if numboynames > numgirlnames:

    print("There are more boy names than girl names with the first letter 'a'.")

elif numboynames < numgirlnames:

    print("There are more girl names than boy names with the first letter 'a'.")

else:

    print("There are equal number of boy names and girl names with the first letter 'a'.")        

        

# What about for every first letter?

import string

for firstletter in string.ascii_lowercase[:26]:

    numboynames = 0

    numgirlnames = 0

    for name in boys.keys():

        if name.startswith(firstletter):

            numboynames = numboynames + 1

    for name in girls.keys():        

        if name.startswith(firstletter):

            numgirlnames = numgirlnames + 1

    if numboynames > numgirlnames:

        print("There are more boy names than girl names with the first letter '%s'." %firstletter)

    elif numboynames < numgirlnames:

        print("There are more girl names than boy names with the first letter '%s'." %firstletter)

    else:

        print("There are equal number of boy names and girl names with the first letter '%s'." %firstletter)      
# I am trying to sum up the values of all boy and girl names.

babiescount = sum(girls.values()) + sum(boys.values())

print(babiescount)
longestname = next(iter(boys)) 

maxlength = len(longestname)

for name in boys.keys():

    if(len(name) > maxlength):

        maxlength = len(name)

        longestname = name

for name in girls.keys():

    if(len(name) > maxlength):

        maxlength = len(name)

        longestname = name

print("The longest name in the dataset is '%s'." %longestname)
num = 0

for name in boys.keys():

    if name in girls.keys():

        num = num + 1

print("There are %s boy names that are also girl names." %num)



num = 0

for name in girls.keys():

    if name in boys.keys():

        num = num + 1

print("There are %s girl names that are also boy names." %num)
count = 0

names = [] #create a list with all boy and girl names (excluding those overlapping ones)

for name in boys.keys():

    names.append(name)

for name in girls.keys():

    if name not in boys.keys():

        names.append(name)



for i in range(1, len(names)):

    for j in range(1, len(names)):

        if names[i] != names[j] and names[i] in names[j]:

            count = count + 1

print("There are %s names that are subsets of other names." %count)
intersect = []

for name in girls.keys():

    if name in boys.keys():

        intersect.append(name)



common = {}

for i in range(1, len(intersect)):

    common[intersect[i]] = girls[intersect[i]]



print("The most popular girl name that is also a boy name is:", max(common, key=common.get))
def babiesnumber(val):

    return boys[val] + girls[val]



val = input("Enter a name: ") 

print("There are %d babies named %s." % (babiesnumber(val), val))
def babiesnumber(val):

    num = 0

    for name in boys.keys():

        if name.startswith(val):

            num = num + boys[name]

    for name in girls.keys():

        if name.startswith(val):

            num = num + girls[name]

    return num



val = input("Enter any prefix: ") 

print("There are %d babies with the prefix '%s' in their names." % (babiesnumber(val), val))
# loading datasets from 2010 to 2013 separately 

NAMES_LIST_2010 = "/kaggle/input/a3data/yob2010.txt"

NAMES_LIST_2011 = "/kaggle/input/a3data/yob2011.txt"

NAMES_LIST_2012 = "/kaggle/input/a3data/yob2012.txt"

NAMES_LIST_2013 = "/kaggle/input/a3data/yob2013.txt"



boys_2010 = {}

girls_2010 = {}

boys_2011 = {}

girls_2011 = {}

boys_2012 = {}

girls_2012 = {}

boys_2013 = {} 

girls_2013 = {}



for line in open(NAMES_LIST_2010, 'r').readlines():  

    name, gender, count = line.strip().split(",")    

    count = int(count)   

    if gender == "F":    

        girls_2010[name.lower()] = count 

    elif gender == "M":

        boys_2010[name.lower()] = count



for line in open(NAMES_LIST_2011, 'r').readlines():  

    name, gender, count = line.strip().split(",")    

    count = int(count)   

    if gender == "F":    

        girls_2011[name.lower()] = count 

    elif gender == "M":

        boys_2011[name.lower()] = count

        

for line in open(NAMES_LIST_2012, 'r').readlines():  

    name, gender, count = line.strip().split(",")    

    count = int(count)   

    if gender == "F":    

        girls_2012[name.lower()] = count 

    elif gender == "M":

        boys_2012[name.lower()] = count

        

for line in open(NAMES_LIST_2013, 'r').readlines():  

    name, gender, count = line.strip().split(",")    

    count = int(count)   

    if gender == "F":    

        girls_2013[name.lower()] = count 

    elif gender == "M":

        boys_2013[name.lower()] = count
# I am defining the most popular names as the name with most babies named in four years in total. Therefore I am summing up the counts and find out the largest ones.



# finding out the most popular girl name

girlsmaxcount = 0

for name in girls_2010.keys():

    if name in girls_2011.keys() and name in girls_2012.keys() and name in girls_2013.keys(): # check if the name is the datasets for all four years

        count = sum([girls_2010[name], girls_2011[name], girls_2012[name], girls_2013[name]])

        if(count > girlsmaxcount):

            girlsmaxcount = count

            populargirlsname = name



print("The most popular girl name across all four years is:", populargirlsname)



# finding out the most popular boy name

boysmaxcount = 0

for name in boys_2010.keys():

    if name in boys_2011.keys() and name in boys_2012.keys() and name in boys_2013.keys(): # check if the name is the datasets for all four years

        count = sum([boys_2010[name], boys_2011[name], boys_2012[name], boys_2013[name]])

        if(count > boysmaxcount):

            boysmaxcount = count

            popularboysname = name



print("The most popular boy name across all four years is:", popularboysname)
# finding out which girl name increased most in popularity and which declined most between 2010 and 2013

girlsmaxdiff = 0

girlsmindiff = 0

for name in girls_2010.keys():

    if name in girls_2013.keys(): # check if the name is in both the 2010 and 2013 datasets

        diff = girls_2013[name] - girls_2010[name]

        if(diff > girlsmaxdiff):

            girlsmaxdiff = diff

            populargirlname = name

        elif(diff < girlsmindiff):

            girlsmindiff = diff

            outtodategirlname = name        

print("Girl names: '%s' has increased most in popularity, and '%s'has declined most in popularity between 2010 and 2013." % (populargirlname, outtodategirlname))



# finding out which boy name increased most in popularity and which declined most between 2010 and 2013

boysmaxdiff = 0

boysmindiff = 0

for name in boys_2010.keys():

    if name in boys_2013.keys(): # check if the name is in both the 2010 and 2013 datasets

        diff = boys_2013[name] - boys_2010[name]

        if(diff > boysmaxdiff):

            boysmaxdiff = diff

            popularboyname = name

        elif(diff < boysmindiff):

            boysmindiff = diff

            outtodateboyname = name        

print("Boy names: '%s' has increased most in popularity, and '%s'has declined most in popularity between 2010 and 2013." % (popularboyname, outtodateboyname))