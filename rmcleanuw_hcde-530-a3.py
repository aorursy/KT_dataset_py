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

        girls[name.casefold()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys[name.casefold()] = count



# We need to format our text so that it can show both the text and an integer.

# But the print() function only takes strings and we have a string and an integer.

# To deal with this, we use the % sign to say what text goes where. In the example 

# below the %d indicates to put a decimal value in the middle of the sentence, and

# that decimal value - the length of 'girls' is indicated at the end of the function: len(girls)



print("There were %d girls' names." %len(girls))

print("There were %d boys' names." %len(boys))



print('There were %d girls\' names' %len(girls) + ' and %d boys\' names.' %len(boys)) #I want to try combining both into a single line. I also want to escape the single quote. 

print('There were %d girls\' names and %d boys\' names.' %(len(girls),len(boys))) #I want to try combining both into a single line using a touple. 



# We did this for you at the end of the previous homework.

# It's a little weird to stuff numbers into sentences this way, but once you get 

# used to it, it's easy. You can do lots of other formatting like this.

# Here's an explanation of how it works: https://www.geeksforgeeks.org/python-output-formatting/
# iterate through the boys dictionary. for each key see if it is: 'john'

for name in boys.keys():

    if name == "john":

        # if it is 'john', get the value associated with john and use that value for the print statement

        # because the value is an integer, we have to cast it to a string in the print statement, with str().

        print("There were " + str(boys[name]) + " boys named " + name.title()) #Add title to make it look more prettier.

        print('There were %d boys named ' %boys[name] + name.title()) #Another way, similar to above.
count = 0

both = []

for name in boys.keys():

    if name in girls.keys():

        count = count+1

        both.append(name)

     

print('There are %d names that show up in both lists.' %count)
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))

import random

random.randint(1,8)
#I know some of the stuff is already done above but I need to start from the top to really understand what is going on.



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

        girls[name.casefold()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys[name.casefold()] = count





#So now I should have boys and girls dictionaries. 



#Count the boys.

boyCountCollector = 0 #Setup a collector and set it to zero.

for nameQty in boys.values(): #Iterate through the items in the dictionary, operating on the value.

    boyCountCollector = boyCountCollector + nameQty #Add the value of each dictionary value to the collector.



#Count the girls.

girlCountCollector = 0

for nameQty in girls.values():

    girlCountCollector = girlCountCollector + nameQty





#Make some output

print('There are %d boys total.' % boyCountCollector)

print('There are %d girls total.' % girlCountCollector)

print('There are %d babies total' %(boyCountCollector+girlCountCollector))
#Let's gather the names into a list.



#We'll start with a blank list.

names = []



#Now let's add all the boy names to the list.

for name in boys.keys():

    names.append(name)



#And let's print the length of the list to help us debug.

print('%d names in the list so far.' %len(names))



#Now let's add all the girl names to the list, if they're not in the list already.

for name in girls.keys():

    if name not in names:

        names.append(name)



print('%d names in the list now.' %len(names))



#Now we can setup nested for loops to go through each name in the list, and compare it to each other name in the list.

#Start with a subset counter set to zero.

subset = 0

#Setup the left side of a comparison using the temporary variable names workingName.

for workingName in names:

    #Setup a breakout counter.

    breakout = 0

    for compareName in names:

        if (subset % 1000 == 0) and (breakout == 0) and (workingName in compareName) and (workingName != compareName):

            print(workingName + " is a subset of " + compareName + "  and there are %d subset names so far" %subset)

            breakout += 1

            subset += 1

        elif (breakout == 0) and (workingName in compareName) and (workingName != compareName):

            breakout += 1

            subset += 1

        elif breakout == 1:

            break



print('%d names are a subset of some other name.' %subset)
#This input works fine in kaggle and you can uncomment it to test, but it's not working when I save the notbook.

#searchName = input("What name would you like to search for? ")



searchName = 'ryan'



#First search the boys dictionary.

for name in boys.keys():

    #If the name is not contained in the list, set the message to 0 and continue on to girls.

    if searchName.casefold() not in boys.keys():

        print("There were 0 boys named "+ searchName.title())

        break

    else:       

        if name == searchName.casefold(): #Using casefold so the input isn't fickle.

            print('There were %d boys named ' %boys[name] + name.title()) #Another way, similar to above.

            

#Now search girls dictionary.

for name in girls.keys():

    #If the name is not contained in the list, set the message to 0 and continue on to girls.

    if searchName.casefold() not in girls.keys():

        print("There were 0 girls named "+ searchName)

        break

    else:       

        if name == searchName.casefold(): #Using casefold so the input isn't fickle.

            print('There were %d girls named ' %girls[name] + name.title()) #Another way, similar to above.