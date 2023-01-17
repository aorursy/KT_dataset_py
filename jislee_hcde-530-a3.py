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



        

if len(girls)>len(boys): #compare how many names are in girls' list with boys list

    print("There are more girls names.")

else:

    print("There are more boys names.")



NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



Names=[]

nameListA = []

nameListB=[]

for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")

    Names.append(name)



for x in Names:

    if x.startswith('A'): #looking for names that start with "A"

        nameListA.append(x) #store the names that start with "A" in the nameListA

    elif x.startswith('B'): #looking for names that start with "B"

        nameListB.append(x) #store the names that start with "B" in the nameListB

print (len(nameListB))

print (len(nameListA))

if len(nameListA)>len(nameListB):  #compare the two lists to see which list has more names

    print ("There are more names start with A than B.")

else:

    print ("There are more names start with B than A.")



NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



COUNT=[]

for line in open(NAMES_LIST, 'r').readlines(): 

    name, gender, count = line.strip().split(",")

    count=int(count)

    COUNT.append(count)

print("Total number of babies born in 2010 is :", sum(COUNT)) # print the sum of all the names in the file from 2010

   
NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



Names=[]

for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")

    Names.append(name)



# Longest String in list 

# using loop 

max_len = -1

for x in Names: #looking for the longest name using a for loop

    if len(x) > max_len: 

        max_len = len(x) 

        longest = x 



# printing result 

print("The longest name is : " + longest) 


babyname=input ("Please enter baby name in lower case letters: ")

for name in girls.keys() & boys.keys(): #find the name in both girls and boys list

    if name==babyname:

        print ("There are "+ str(girls[name])+ " girls named "+ babyname) #print how many girls which that name

        print ("There are "+ str(boys[name])+ " boys named "+ babyname) #print how many boys which that name
