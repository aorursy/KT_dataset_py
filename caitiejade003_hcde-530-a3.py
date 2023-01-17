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

    if name == "joseph":

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

if len(girls) > len(boys):   #compare the length of the dictionaries created in example 1

    print("There are more girls names than boys names")

else:

    print("There are more boys names than girls names")

    

    

#compare two letters to the first letters of each name

x = "c"  #use variables to enter letters

y = "r"   #use variables to enter letters

how_manyx = 0  #for counting

how_manyy = 0





for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")

    name = name.lower()  #make it all lower case

        

    if x in name[0]:       #see if x is the in the first position of the string

        how_manyx = how_manyx + 1  #if so add to the counter

        

    elif y in name[0]:    #see if x is the in the first position of the string

        how_manyy = how_manyy + 1   #if so add to the counter



print("There are " + str(how_manyx) + " names that start with " + x + " and " + str(how_manyy) + " names that start with " + y)        

        

if how_manyx > how_manyy:

    print("More names start with the letter " + x + " than the letter " + y)

else:

    print("More names start with the letter " + y + " than the letter " + x)

        

        
#Create a variable to hold the counter

babies = 0



#Then I'll open the file in read mode

babylist = open("/kaggle/input/a3data/yob2010.txt", "r") 



#Do a for loop to look at each line

for line in babylist:

    name, gender, count = line.strip().split(",") #split line into 3 variables so the number of babies with that name/gender is on it's own

    count = int(count)  #ensure count variable is integer 

    babies = babies + count #add that number to the babies variable

    

print("There were %d babies in the dataset." %babies) #show the total



babylist.close()
#create an empty list

all_names = []

#create placeholder for length of name

namelength = 0

#create placeholder for longest name

longest_name = "name"



#Open the file in read mode

babylist = open("/kaggle/input/a3data/yob2010.txt", "r") 



#Do a for loop to look at each line

for line in babylist:

    name, gender, count = line.strip().split(",") #split line into 3 variables

    all_names.append(name) #add the name to the all names list

    

for item in all_names: #iterate through the list of names

    if len(item) > namelength: #compare the length of each name to the namelength variable

        namelength = len(item) #if it is longer, update namelength variable

        longest_name = item #also update longest name variable

print ("The longest name is %s." %longest_name)    #once the program looks through all the names, the longest one is stored in longest_name and can be printed



babylist.close()  #close the file
#Final

x = input("Type a name to look it up: ").lower() #assign an input to a variable and made sure it would work if entered with lower or upper case

total_count = 0 #create variable to hold number



for key in boys: #iterate through boy dictionary previously created in example 1

    if key == x: #check if it has a key equal to the input

        print("There were " + str(boys[x]) + " boys named " + x.capitalize()) #print out value of how many boys

        total_count = total_count + boys[x]  #add the number to the total count



for key in girls:  #iterate through girl dictionary previously created in example 1

    if key == x:  #check if it has a key equal to the input

        print("There were " + str(girls[x]) + " girls named " + x.capitalize()) #print out value of how many girls

        total_count = total_count + girls[x]  #add the number to the total count

        

print("There were %d babies total with that name" %total_count)        #tell user the total count

        
#3rd try for #7: I looked through the boys and girls lists separately and printed out the number separately

#I still wanted to add them both together and also realized there was also a problem with capitalization



x = input("Type a name to look it up ") #assign an input to a variable



if x in boys.keys(): #reusing the dictionaries created above for this one

 # if it is same as the name the user input print the value

    print("There were " + str(boys[x]) + " boys named " + x)

else:

    print("There were no boys with this name")  

    

if x in girls.keys():

 # if it is same as the name the user input print the value    

    print("There were " + str(girls[x]) + " girls named " + x)

else:

    print("There were no girls with this name")
#2nd try for #7 I made a single list, but realized it still only counted one version of the name if it was used for both boys and girls



dict_names = {}  # create an empty dictionary of key:value pairs



#Open the file in read mode

babylist = open("/kaggle/input/a3data/yob2010.txt", "r") 



for line in babylist:

    name, gender, count = line.strip().split(",") #split line into 3 variables

    dict_names[name.lower()] = count #stores the name as the key and the count as the value in the dictionary



x = input("Type a name to look it up ") #assign an input to a variable



#reusing the dictionaries created above for this one

if x in dict_names.keys():      #looks for the name in the dictionary

    # if it is same as the name the user input

    print("There were " + str(dict_names[x]) + " children named " + x)





#1st try for #7 I looked through each list, but if the name was used on both boys and girls, it only displayed the boys count

#if x in boys.keys(): #reusing the dictionaries created above for this one

    # if it is same as the name the user input

#    print("There were " + str(boys[x]) + " boys named " + x)

#elif x in girls.keys():

#    print("There were " + str(girls[x]) + " girls named " + x)

#else:

#    print("This name is not on the list")

        
boys2010 = {}  # create an empty dictionary of key:value pairs for the boys

girls2010 = {} # create an empty dictionary of key:value pairs for the girls



boys2011 = {}  # create an empty dictionary of key:value pairs for the boys

girls2011 = {} # create an empty dictionary of key:value pairs for the girls



boys2012 = {}  # create an empty dictionary of key:value pairs for the boys

girls2012 = {} # create an empty dictionary of key:value pairs for the girls



boys2013 = {}  # create an empty dictionary of key:value pairs for the boys

girls2013 = {} # create an empty dictionary of key:value pairs for the girls



#trying out the with open statement

with open("/kaggle/input/a3data/yob2010.txt", "r") as file_2010, open("/kaggle/input/a3data/yob2011.txt", "r") as file_2011, open("/kaggle/input/a3data/yob2012.txt", "r") as file_2012, open("/kaggle/input/a3data/yob2013.txt", "r") as file_2013:



#making all the dictionaries separately

    for line in file_2010:

        name, gender, count = line.strip().split(",") #split lines of 2010 list into 3 variables

        

        count = int(count)   # Cast the string 'count' to an integer

    

        if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

            girls2010[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

        elif gender == "M": # Otherwise store it in the boys dictionary

            boys2010[name.lower()] = count

            

    for line in file_2011:

        name, gender, count = line.strip().split(",") #split lines of 2011 list into 3 variables

        

        count = int(count)   # Cast the string 'count' to an integer

    

        if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

            girls2011[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

        elif gender == "M": # Otherwise store it in the boys dictionary

            boys2011[name.lower()] = count

            

    for line in file_2012:

        name, gender, count = line.strip().split(",") #split lines of 2012 list into 3 variables

        

        count = int(count)   # Cast the string 'count' to an integer

    

        if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

            girls2012[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

        elif gender == "M": # Otherwise store it in the boys dictionary

            boys2012[name.lower()] = count

            

    for line in file_2013:

        name, gender, count = line.strip().split(",") #split lines of 2013 list into 3 variables

        

        count = int(count)   # Cast the string 'count' to an integer

    

        if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

            girls2013[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

        elif gender == "M": # Otherwise store it in the boys dictionary

            boys2013[name.lower()] = count



            

#Combine the dictionaries,trying this Counter function

from collections import Counter



girls_allyears = Counter(girls2010) + Counter(girls2011) + Counter(girls2012) + Counter(girls2013)  #new dictionary adding all years of girls

boys_allyears = Counter(boys2010) + Counter(boys2011) + Counter(boys2012) + Counter(boys2013)       #new dictionary adding all years of boys



#Now go through the complete dictionaries to find highest number



fbabyname_count = 0 # variable to compare the value 

fmost_popular = ""   # variable to store a name



mbabyname_count = 0 # variable to compare the value 

mmost_popular = ""  # variable to store a name

###

for x in girls_allyears: #iterate through the combined girls dictionary

    if girls_allyears[x] > fbabyname_count: #compare the value to the counter variable

        fbabyname_count = girls_allyears[x] #if it is longer, update variable

        fmost_popular = x #also update longest name variable

print("The most popular girl name was %s with %d babies." % (fmost_popular, fbabyname_count))        





for x in boys_allyears: #iterate through the combined boys dictionary

    if boys_allyears[x] > mbabyname_count: #compare the value to the counter variable

        mbabyname_count = boys_allyears[x] #if it is longer, update variable

        mmost_popular = x #also update longest name variable

print("The most popular boy name was %s with %d babies." % (mmost_popular, mbabyname_count))    





            