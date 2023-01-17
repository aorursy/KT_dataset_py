NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls

both = {}



for line in open(NAMES_LIST, 'r').readlines():

    name, gender, Count = line.strip().split(",")  

    Count = int(Count)   # Cast the string 'count' to an integer

    

    if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

        girls[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys[name.lower()] = count

    #if gender == "F" or "M":

    #   both[name.lower()] = count

        



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

#1. Are there more boy names or girl names? What about for particular first letters? What about for every first letter?



#use boolean to compare number of girls and boys name and print out conditions accordingly

if len(girls) > len(boys):

    print("There are more girl's names")

elif leng(girls) < len(boys):

    print("There are more boy's names")

    



#2. How many babies are in the dataset (assuming nobody is counted more than once)?

#answer should be 3684761



#open file

fh = open("/kaggle/input/a3data/yob2010.txt", 'r')



#create variable that will count total babies

totalbaby = 0



#for every 'baby' in file print out count of each name

for baby in fh:

    baby = baby.strip().split(',')

    totbab = baby[2] #this pulls second index from each line, which is the count

    #print(baby[2]) #test to see if count is printed correctly for each name

    totalbaby = int(totbab) + totalbaby #convert totbab into an integer and add count that is next in line



print(totalbaby)

fh.close()
#3. What is the longest name in the dataset?

babyfile = open("/kaggle/input/a3data/yob2010.txt", 'r')



max_so_far = 0



for line in babyfile:

    line = line.strip().split(',')

    babyname = (line[0])

    babynamelength = len(line[0])

    intbabynamelength = int(babynamelength)

    if intbabynamelength > max_so_far:

        max_so_far = intbabynamelength



print("the longest baby name has %d characters" %max_so_far)



babyfile.close()

#3. What is the longest name in the dataset?



babyfile = open("/kaggle/input/a3data/yob2010.txt", 'r')



babyname = (line[0])

babynamelength = len(line[0])



for line in babyfile:

    line = line.strip().split(',')

    intbabynamelength = int(babynamelength)



print(intbabynamelength) + print(babyname)



    





fh.close()

#4. How many boy names are also girl names? How many girls' names are also boys' names?

boysgirlname = 0

girlboysname = 0



for name in boys.keys():

    if name in girls.keys():

      boysgirlname = boysgirlname + 1



    

for name in girls.keys():

    if name in boys.keys():

      girlboysname = girlboysname + 1



print('%d boy names are also girl name' %boysgirlname)

print('%d girl names are also boy name' %girlboysname)
