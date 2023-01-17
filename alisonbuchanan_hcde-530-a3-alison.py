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
#3. What is the longest name in the dataset?



#open the entire names file so we can work with all of the names 

NAMES_LIST = open("/kaggle/input/a3data/yob2010.txt")



#declare varible to use for storing results after each iteration through the data

length = 0

longest_name = ()



for line in NAMES_LIST: # foor loop set up so we can iterate through each line

    name, gender, count = line.strip().split(",") #split the line into name, gender, and count variables

    #this isolates the name varible, which is the only variable we are interested in for anwering this question 

    if (len(name)>length): # check the number of characters in the name variable

        #if the number of characters in the next name is bigger than or equal to the last name we stored

        longest_name = name #update the longest name variable to the last name found, which is now the longest

        length = len(name) #update length to equal the length of the current longest name

    #but what if there are several "longest names" that are all the same length? We want to know about those!      

    if (len(name) == length): # if the name lengths equal eachother 

        longest_name = name + " " + longest_name #update longest name to print both equally long names

        length = len(name)  #update length to equal the length of the current longest name(s)

              

print(longest_name) #print out the longest name (s)
#4. How many boy names are also girl names? How many girls' names are also boys' names?



#declare variables that will be used to hold totals/count things later on in the code

gtotal = 0

btotal = 0



#split lines in text file to separate our name, gender and count info 

name, gender, count = line.strip().split(",")



#how many boy names are also girl names?

for name in boys.keys(): #iterate over all names in the boy name list

    if name in girls.keys(): #compare variable 'name' in boy list to girl list. If 'name' is also in girl list...

        # add one to total number of names that apear in boy and girl list 

         gtotal = gtotal + 1



print("%d boys' names are also girls' names." %gtotal) #print the final total 



#how many girl names are also boy names - this will give us the same answer as the above code, but it is fun to do it this way

for name in girls.keys(): #iterate over all names in the girl name list

    if name in boys.keys(): #compare variable 'name' in boy list to girl list. If 'name' is also in girl list...

        # add one to total number of names that apear in boy and girl list 

        btotal = btotal + 1



print("%d girls' names are also boys' names." %btotal) #print the final total 
# 6. What is the most popular girl name that is also a boy name?



#declare variables that will be used to count things/store items as we iterate through things later

last_count = 0

popular_name = () 



for name in boys.keys(): #iterate over all names in the boy name dictionary

    if name in girls.keys(): #compare variable 'name' in boy dictionary to girl dictionary. 

        #If 'name' is also in girl dict....

        if girls[name] > last_count:# check if current count variable of the name is greater than last count we stored

            # we are interested in the count variable because it tells us how popoular the name is

            popular_name = name #if the current name has the highest count, store it as the most popular name

        #but what if there are two names that have equally high counts/popularity? 

        if girls[name]== last_count: # check if current count variable of the name is the same as the last count we stored 

            popular_name = name + " " + popular_name #if the current name is equal to the last highest count

            # store both the former popular name and the new popuar in the variable popular_name

            #becuase they are both the most popular name

            #this way when we print popular name, it will print out the two, equally popular names

print("the most popular girl name that is also a boy name is "+ popular_name)