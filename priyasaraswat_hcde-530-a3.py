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

Total_girls_names = len(girls)

Total_boys_names = len(boys)

#The below if compares the total key values in both girls and boys dictionary

if(Total_girls_names > Total_boys_names):

    print("There are more girl names than boy names")

else:

    print("There are more boy names than girl names")



print("-------------------------------------")



#Identifying gender neutral name by comparing key values in both dictionary and storing the 

#count value for common names

common_count = 0

for name in boys.keys():

    if name in girls.keys():

        common_count = common_count + 1



#Performing calculation based on unique names in both dictionaries

print("Total gender neutral names = ", common_count)      

Total_unique_girls_names = len(girls) - common_count

Total_unique_boys_names = len(boys) - common_count



print("Total unique girls names = ", Total_unique_girls_names)

print("Total unique boys names =", Total_unique_boys_names )



if(Total_unique_girls_names > Total_unique_boys_names):

    difference = Total_unique_girls_names - Total_unique_boys_names

    print("There are",difference,"more unique girls names than unique boys names")

else:

    difference = Total_unique_boys_names - Total_unique_girls_names

    print("There are more unique boys names than girls names")

    
#Asking user to input few letters

print("Enter few letters to search as first letters in the dictionary")

print("--------------------------------------")

first_letter = input("Enter few characters : ") 



#Checking in girls dictionary

girls_with_first_letters = 0

for name in girls.keys():

    #Checking if their is matching substring

    if(name.find(first_letter) != -1):

        value = name.find(first_letter)

        #Checking if position is at the beginning of each name

        if (value == 0):

            girls_with_first_letters = girls_with_first_letters + 1

#Printing total girls names with first letters

print("Total girls names with first letters", first_letter,"–", girls_with_first_letters)



#Checking in boys dictionary

boys_with_first_letters = 0 

for name in boys.keys():

    #Checking if their is matching substring

    if(name.find(first_letter) != -1):

        #Checking if position is at the beginning of each name

        value = name.find(first_letter)

        if (value == 0):

            boys_with_first_letters = boys_with_first_letters + 1

#Printing total boys names with first letters

print("Total boys names with first letters", first_letter,"–", boys_with_first_letters)



#Checking if there are more boy names or girl names starting with the letters entered by user

if(Total_girls_names > Total_boys_names):

    print("There are more girls names starting with",first_letter,"than boys names")

else:

    print("There are more boys names with",first_letter," than girls names")
#new empty girls names dictionary

gname_letters = {}



#sorted the girls names in alphabetical order

sorted_girls = sorted(girls.keys(), key=lambda x: x[0])

for name in sorted_girls:

    #Extracting the first letter from the name 

    starts_with = name[0]

    #Checking if the first letter is a key in new girls names dictionary increment the counter

    if starts_with in gname_letters:

        gname_letters[starts_with]+= 1

    #else starts the counter over

    else:

        gname_letters[starts_with] = 1

#Alphabetically sorted girls names list is ready

#print(gname_letters)



#new empty boys names dictionary

bname_letters = {}



#sorted the boys names in alphabetical order

sorted_boys = sorted(boys.keys(), key=lambda x: x[0])

for name in sorted_boys:

     #Extracting the first letter from the name 

    starts_with = name[0]

    #Checking if the first letter is a key in new girls names dictionary increment the counter

    if starts_with in bname_letters:

        bname_letters[starts_with]+= 1

    #else starts the counter over

    else:

        bname_letters[starts_with] = 1

        

#Alphabetically sorted boys names list is ready

#print(bname_letters)



#Comparing the total name count for each letter in both the lists and printing the outcome 

for count in gname_letters.keys():

    if(gname_letters[count] > bname_letters[count]):

        print('For names starting with', count, ': Girls names are more than boys name')

    else:

        print('For names starting with', count, ': Boys names are more than girls name')

    



total_babies = 0



#looping girls list to get total count

for name in girls.keys():

    total_babies = girls[name] + total_babies



#looping boys list to get total count

for name in boys.keys():

    total_babies = boys[name] + total_babies



#Printing total babies in the dataset

print("Total number of babies in dataset :",total_babies)

   
count = 0

#Looping girl's name

for name in girls.keys():

        #Storing the length of string in variable count and comparing it with the new length

        #if count is smaller than the new length, count is replaced with new length value

        if len(name) > count: 

           count = len(name)

           #Word will store the name with longest length 

           word = name

print("The longest girl name with %d character is" %len(word), word)



#Looping boy's name

for name in boys.keys():

        #Storing the length of string in variable count and comparing it with the new length

        #if count is smaller than the new length, count is replaced with new length value

        if len(name) > count: 

           count = len(name)

           #Word will store the name with longest length

           word = name

print("The longest boy name with %d character is" %len(word), word)



print("The longest name therefore, is –", word)
#counter for total boy names are also girl names

names_count = 0

for name in boys.keys():

    if name in girls.keys():

        names_count+= 1

print("Total boy names that are also girl names:", names_count)



#counter for total girl' names that are also boys' names

count = 0

for name in boys.keys():

    if name in girls.keys():

        count+= girls[name]



print("Total boys'names that are also girls'names:", count)

total_names = 0

#Empty list to store all names values

x = []

#To add to the list of unique boys names

for name in boys.keys():

    if name in girls.keys():

        continue

    else:

        x.append(name) 

        

#To add to the list of unique girls names

for name in girls.keys():

    if name in boys.keys():

        continue

    else:

        x.append(name)  

        

#To add to the list of common names

for name in boys.keys():

    if name in girls.keys():

        x.append(name)         

        
#First for loop to pick one name from the list

for name in x:

    #Second for loop to compare the picked name with all the other names in the list

    for i in x:

        #Ignoring the name if the two are same 

        if i == name:

            continue

        #Ignoring the name if not able to find a substring 

        elif i.find(name) == -1:

            continue

        #Incrementing the counter for rest of the cases

        else:

            count+= 1

    #Checking if the counter is not zero which means no subset found, otherwise

    #incrementing the number of names that are subset of other names

    if count!= 0:

        total_names+=1



print(f"{total_names} names are subset of {count} names")