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

# function to count the names in the dictionary

# Accepts dictionary to work with and the letters the name starts with

# In multiple letters are provided in startsWith, the function matches the whole substring, i.e. Jo >> John

# If startsWith="*", the functon will return the length of the dictionary

# Function returns the number of names in the dictionary, which matched provided pattern

def countByStartsWith (dictionary, startsWith):

    if startsWith=="*":

        return len(dictionary)

    count = 0

    for i in dictionary.keys():

        if i[:len(startsWith)].lower()==startsWith.lower():

            count+=1

    return count



# One more function to calculate and output the results

# Accepts boy and girl dictionaries and the letter the name starts with (or * for any letter)

# Calculates and prints the results

def compareNames (boysDic, girlsDic, startsWith):

    totalBoys = countByStartsWith(boysDic, startsWith)

    totalGirls = countByStartsWith(girlsDic,startsWith)

    if startsWith=="*":

        startStr = "in the dictionary."

    else:

        startStr = "starting with "+startsWith.upper()+"."

    print ("There are", totalBoys,"boy's and", totalGirls, "girl's names", startStr)

    if totalBoys>totalGirls:

        print ("There are more BOY'S names "+startStr)

    elif totalBoys==totalGirls:

        print ("There is equal number of boy's and girl's names "+startStr)

    else:

        print ("There are more GIRL'S names "+startStr)

    

#now let's have fun counting

compareNames(boys, girls, "*")

print("************")

#let us iterate through the alphabet

for letter in 'abcdefgijklmnopqrstuvwxyz':

    compareNames (boys, girls, letter)

    print("************")

    
totalBoys = 0

for name in boys.keys():

    totalBoys+=boys[name]

totalGirls = 0

for name in girls.keys():

    totalGirls+=girls[name]

print("There are",totalBoys,"baby boys and", totalGirls, "baby girls in the dataset.",totalBoys+totalGirls,"babies in total.")
allNames = {}

allNames.update(boys) #lump boys into common list

allNames.update(girls) #lump girls into common list

longestName = ""

for name in allNames.keys(): #iterate through common list

    if len(name)>len(longestName):

        longestName=name

print("The longest name in the dataset is ",longestName.title(),", and it is",len(longestName),"characters long.")
#This is one way to solve this problem

print ("Simple math tells us that there are ",len(boys)+len(girls)-len(allNames), "boys' names that are also girls' names and the same number of girls' names that are also boys' names")



#But, since we are going to need the list of names common between boys and girls later, here is another solution

unisex = []

for name in boys.keys():

    if name in girls.keys():

        unisex.append(name)

print ("If we look closer, we see",len(unisex), "names common across boys and girls. Here they are:")

i=0

while i<=len(unisex):

    print(unisex[i:i+10])

    i+=10

counter = 0 #no names found yet

listOfNames = list(allNames.keys()) #get the keys out of the dictionary and put them into the list



j=0 #start with the first name

l=len(listOfNames) #this is the length of the list, lets keep it in a variable, we are going to need it a lot



while j<l: 

    i=j+1 #compare current name starting from the next in the list

    while i<l:

        if listOfNames[j] in listOfNames[i]:

            counter+=1 #subset found

            i=l #skip to the next name

        else:

            i+=1 #subset is not found, contine to the next pair

    j+=1 #repeat for the next name in the list

print (counter,"out of", len(listOfNames), "names are subsets of other names")

   
topName=unisex[0] #we start with the first name

for name in unisex: #and iterate through the list

    if girls[name]>girls[topName]: #comparing the popularity of current name to the most popular so far

        topName=name

print ("The most popular girl's name among unisex names is",topName.title()+".", "It was used", girls[topName], "times. Can you imagine that", boys[topName],"families named their baby boy",topName.title()+"?!")

topName=unisex[0]

for name in unisex:

    if boys[name]>boys[topName]:

        topName=name

print ("Maybe they are friends of",girls[topName],"families, who named their baby girl",topName.title()+"?", "That was the most popular boy's name in the list of unisex names")
print("Please input the name:")

theName = input().strip().lower()

print("There were",boys.get(theName,0),"boys and",girls.get(theName,0),"girls named",theName.title())

# function to count the babies by names in the dictionary

# Accepts dictionary to work with and the letters the name starts with

# In multiple letters are provided in startsWith, the function matches the whole substring, i.e. Jo >> John

# Function returns the number of babies, with names matching provided pattern

def babiesByStartsWith (dictionary, startsWith):

    count = 0

    for i in dictionary.keys():

        if i[:len(startsWith)].lower()==startsWith.lower():

            count+=dictionary[i]

    return count



#now let's accept the pattern

print("The name starts with...?")

pattern = input().strip().lower()

print("There were",babiesByStartsWith(boys, pattern),"boys and",babiesByStartsWith(girls, pattern),"girls with names starting with",pattern.title())