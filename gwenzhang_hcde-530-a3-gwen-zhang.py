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

#iterate through all the keys in dictionary boys to check if any boys name is also in 
# the keys list of girls dictionary, if so, print out the name.
for name in boys.keys():
    if name in girls.keys():
        print(name)
#iterate through the key list of the boys dictionary
for name in boys.keys():
    #if 'king' substring inside the name, print out the name.
    if 'king' in name:
        print(name + " " + str(boys[name]))
#iterate through the key list of the girls dictionary
for name in girls.keys():
    #if 'queen' substring inside the name, print out the name.
    if 'queen' in name:
        print(name + " " + str(girls[name]))

#2. How many babies are in the dataset (assuming nobody is counted more than once)?
# sum up value of each keys mapping to for both boys and girls dictionary
# which result is the totaly number of babies.
#use result to store the current total number.
result = 0
for name in boys.keys():
    #add up number of babies of each name and store it to result.
    result = result + boys[name]

for name in girls.keys():
    #same as above
    result = result + girls[name]
#print out result number.
print("There are " + str(result) +" babies in the dataset." )
#3. What is the longest name in the dataset?
#use a variable called longetname to store the longest name.
longestname = ""
#iterate through the key list of the boys dictionary
for name in boys.keys():
    #whenever found a name is longer than current longestname, 
    # we store the new longest name in longestname.
    if len(name)>len(longestname):
        longestname = name
# go through the same process for girls name
for name in girls.keys():
    if len(name)>len(longestname):
        longestname = name
#At last print out the longest name among boys and girls.
print(longestname)
        
#4. How many boy names are also girl names? How many girls' names are also boys' names?
# use count2 work as a counter.
count2 = 0
for name in boys.keys():
    #if a name in boys key list also in girls key list, we
    # add 1 to the counter 
    if name in girls.keys():
        count2 = count2 + 1
print(count2)

count3 = 0
for name in girls.keys():
    if name in boys.keys():
        count3 = count3 +1
print(count3)
#9. Which boy and girl names are the most popular across all four years in our dataset?

#we form a list of text names so we can do same action accros all files.
NAMES_LISTS = ["/kaggle/input/a3data/yob2010.txt", "/kaggle/input/a3data/yob2011.txt","/kaggle/input/a3data/yob2012.txt","/kaggle/input/a3data/yob2013.txt"]

boysTotal = {}  # create an empty dictionary of key:value pairs for the boys
girlsTotal = {} # create an empty dictionary of key:value pairs for the girls
# iterate throgh all files.
for NAMES_LIST in NAMES_LISTS:
    # read each lines in file NAMES_List
    for line in open(NAMES_LIST, 'r').readlines():  
        #assign the first, second and third element to name, gender, count variables.
        name, gender, count = line.strip().split(",")
        # cast the count from string type to integer type
        count = int(count)      
        # if the gender is a female we process the following logic
        if gender == "F":
            #if the name has been store in the girlsTotal dictionary,
            # we fetch the value by key and add up the value to that key.
            if name.lower() in girlsTotal:
                girlsTotal[name.lower()] = girlsTotal[name.lower()] + count
            # if the key never get store in the dictionary, we add the key
            # value pair to the dictionary.
            else:
                girlsTotal[name.lower()] = count 
         # if the gender is a male we process the following logic same as above.       
        elif gender == "M": 
            if name.lower() in boysTotal:
                boysTotal[name.lower()] = boysTotal[name.lower()] + count
            else:
                boysTotal[name.lower()] = count 

popularBoyName = ""  # place holder for most popular boy's name
boyNameCount = 0     # count place holder for the popular boy name amount.
popularGirlName = "" # place holder for most popular girl's name
girlNameCount = 0    # count place holder for the popular girl name amount.

#iterate through the boys name and amount dictionary.
# find the most popular boys name.
for boyname in boysTotal.keys():
    if boysTotal[boyname] > boyNameCount:
        boyNameCount = boysTotal[boyname]
        popularBoyName = boyname
#iterate through the girl's name and amount dictionary.
# find the most popular girl's name.       
for girlname in girlsTotal.keys():
    if girlsTotal[girlname] > girlNameCount:
        girlNameCount = girlsTotal[girlname]
        popularGirlName = girlname
#print out the result.
print(popularBoyName, popularGirlName)