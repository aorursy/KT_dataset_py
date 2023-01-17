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

#First I assign the value 0 to the variables, since it is the running total of the values so far. 



totalBoys = 0

totalGirls = 0



#Next I create a for loop to iterate the names in the boys dictionary.

for name in boys.keys():

    #I use this function to add the number of boys names to the value of the totalBoys variable.

    totalBoys += boys[name]



#I also create one for the names in the girls dictionary.

for name in girls.keys():

    #I use this function to add the number of girls names to the value of the totalGirls variable.

    totalGirls += girls[name]



#Then I print some text to make the output fit in a sentence. I also print the indivual total of boy names and girl name, then add them up using the + in the print function. 

print ("There are", totalBoys, "boy babies and", totalGirls, "girl babies, for a combined total of", totalBoys+totalGirls, "total babies in the dataset.")
#First I assign the value 0 to the variable, a running total of the values to far. In this exercise I'm using:

longestName = 0



#Next I start by creating a for loop to iterate the names in the boys dictionary. 

for name in boys.keys():

    #The if statement is used to indicate that if the length of the name is greater than the previous name, it becomes the new longest name.

    if len(name) > longestName:

        #here i assign the variable to the value of the longest name 

        longestName = len(name)

        #here i assign the variable with the longest name to the actual name of that value.

        longestBoy = name

#then i print the actual name and length. 

print ("The longest boy name is", longestBoy, "with a total of", longestName, "letters.")



#when I first did this part, it kept giving me an error that longestGirl was not defined, had a hard time figuring this out. I found that I had to assign the longestName variable to 0 again before creating the new for loop for girls. So this is after many attemps at figuring it out lol.  

longestName = 0

#basically, I repeated here what I did with the for loop for boys above. 

for name in girls.keys():

    if len(name) > longestName:

        longestName = len(name)

        longestGirl = name

#printing the actual name and its length. 

print ("The longest girl name is", longestGirl, "with a total of", longestName, "letters.")



#since the longest name is the boy name with 15 letters, I print that name as the final answer. 

print ("The longest name in the dataset is", longestBoy, ".")
#assign variable the value 0, the running total of values for this variable and to access it in the for loop. 

popularGirl = 0



#next I create a for loop to iterate through names in the girls dictionary.

for name in girls.keys():

    

    #then i created a function where if the girls name is greater than the original variable popularGirl, it becomes the new most popular name. I also used the "and" function to filter out names that are not in the boys dictionary. 

    if girls[name] > popularGirl and name in boys.keys():

        popularGirl = girls[name]

        

        #at first when I tried printing, I had the print function aligned all the way to the left and kept getting a name that was definitely not popular, "zyrihanna". Then I figured out that I had to move the print function to align with the last line in the block above. Had a hard time realizing that but felt silly after I did figure it out. 

        print ("The most popular girl name that is also a boy name is", name, ".")