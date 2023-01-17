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

        print("There were " +str(boys[name]) + " boys named " + name) 

for name in boys.keys():

    if name in girls.keys():

        print(name)
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))

#6. What is the most popular girl name that is also a boy name?



#an accumulation pattern to find both the most popular girl and double check that this girl is also a boys name

#maxnum for checking popularity

maxNum = 0

#declaring the empty string 'mostPopularGirl'

mostPopularGirl = ""



#if name is in both girl and boys

for name in girls.keys():

    if name in boys.keys():



#accumulation pattern to find the largest count for girl

        num = girls[name]

        if num > maxNum:

            

#set the largest num counting name to mostPopularGirl

            maxNum = num

            mostPopularGirl = name

print(mostPopularGirl)

            

            





    

    



#2. How many babies are in the dataset (assuming nobody is counted more than once)?





#an accumulation pattern to find the total number of babies in the set

#start the count at 0 for girls names

countG = 0

#start the count at 0 for boys names

countB = 0



#loop through the girls dictionary and count all the girls

for count in girls.keys(): 

    countG = girls[count] + countG

    

#loop through the boys dictionary and count all the boys

for count in boys.keys(): 

    countB = boys[count] + countB    

    

#print the total count of girl babies

print("There are " + str(countG) + " girl babies")



#print the total count of boy babies

print("There are " + str(countB) + " boy babies")



#print the total count of boy + girl babies

print("There are " + str(countG + countB) + " total babies")







            

#3. What is the longest name in the dataset?



#set a counter nameNum to 0

nameNum = 0

#declaring the empty strings for 'longestName' of boys and girls 

longestNameG = ""

longestNameB = ""



#loop through the girls dictionary

for name in girls.keys():

    #find the longest name 

    if len(name) > nameNum:

        #set my counter to the length of the longest name

        nameNum = len(name)

        #give me the name for that longest name value at store in longestNameG

        longestNameG= name

        

#loop through the boys dictionary

for name in boys.keys():

    #find the longest name 

    if len(name) > nameNum:

        #set my counter to the length of the longest name

        nameNum = len(name)

        #give me the name for that longest name value at store in longestNameB

        longestNameB= name



#compare the length of the strings longestNameG and longestNameB

#print the larger string

if len(longestNameG) > len(longestNameB):

    print (longestNameG)

else:

    print (longestNameB)




