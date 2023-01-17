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

if len(girls)> len(boys):

    print("There are more girls names than boys names")

else:

    print("There are more boys names than girls names")



kgirlnames={}#Define a dictionary to store the girls names that start with K in it

for letters in girls:

    if letters[0]=="k":#Search for names that start with the letter K in the list of girls names

        kgirlnames[letters] = str #Store those names in the dictionary and classify them as Strings

#Same process below for boys names

kboynames={}

for lettersb in boys:

    if lettersb[0]=="k":

        kboynames[lettersb] = str

    

if len(kboynames)>len(kgirlnames):#Determine which group is larger and print the result

    print("There are more boys than girls that have names that start with K")

else:

    print("There are more girls than boys that have names that start with K")

    




allnames={}#Start a new dictionary for all of the names together

largestword=0#Create a variable for the longest name

readit = open("/kaggle/input/a3data/yob2010.txt", 'r')

for long in readit:

    name, gender, count = long.split(",")#Loop the file and divide the 3 categories up by their split on ","

    allnames[name.lower()] = count#Store the name category in the allnames dictionary

for longest in allnames.keys():#Loop new dictionary

    if largestword<(len(longest)):

        largestword=(len(longest))#Replace variable largestword with the largest integer that it loops on

    if (len(longest))==15:#I have already discovered that the longest word is 15 letters long, and worked backwards at this point. I am now looking for all of the names which are 15 letters long. 

        print("The longest names are :%s"%longest)

print("The longest name is %s" %largestword + " letters long")#Identify the length of the longest word







thesum = 0#Create a variable for the sum

readit = open("/kaggle/input/a3data/yob2010.txt", 'r')

for total in readit:

    name, gender, count = total.split(",")

    total.strip()

    count=int(count)#Cast the variable count as an integer

    thesum=(thesum+count)#Create a formula to add each number to the next

print("There were %d babies in the dataset" %thesum)
check = input("Write a name in to find out how many babies were born with than name in 2010:")

#Write the input question

if check in (allnames.keys()):#Write a logic to investigate if the input exists in the dictionary

    print(allnames[check])#Output the value of the input if it is in the dictionary

else:

    print("No one was born with that name")#Create alternative if the name is not in the dictionary