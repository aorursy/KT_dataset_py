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

        print(name, len(name))
for name in boys.keys():

    if 'joe' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'joe' in name:

        print(name + " " + str(girls[name]))

total_girls = 0 

total_boys  = 0



#empty variables to store the total baby numbers



for g in girls.keys():

    total_girls += girls[g]

    

#the variable g loops through girls adding to the count which is stored in the total_girls variable



for b in boys.keys():

    total_boys += boys[b]

    

#the variable b loops through boys adding to the count which is stored in the total_boys variable



total_babies = total_girls + total_boys



#both total variables are added together



print("There are " + str(total_babies) + " total babies.")



#a string is printed out using the total of total variables

boy = 0



#empty variable to count total boys



for name in boys.keys():

    if name in girls.keys():

        boy += 1

        

#loop saying if the name is in boys and also in girls, add a count to the boy variable

        

print("There are " + str(boy) + " total boy names that are also girl names.")



#prints a sentence containing the total ammount boy variable



girl = 0 



#empty variable to count total girls



for nn in girls.keys():

    if nn in boys.keys():

        girl += 1

        

#loop saying if the name is in girls and also in boys, add a count to the girl variable

        

print("There are " + str(girl) + " total girl names that are also boy names.")



#prints a sentence containing the total ammount girl variable
popular_count = 0 



#empty variable to hold the total count



for name in girls.keys():

    if name in boys.keys():

        count = girls[name]

        

#loop saying if the name is in girls, and then also in boys add a count to the variable count

        

        if count > popular_count:

            popular_count = count

            popular_name = name

            

#if the count variable from the loop above is greater than the value of the original popular_count variable, updated the popular_count variable to that count and record the name. 

print(popular_name)



#print the popular name

        





       

n = "isabella"



#n is equal to the changable variable, update as you like! 



for name in boys.keys():

    if name == n:

        print("There were "+ str(boys[name]) + " boys named " + n)

        

#for loop that goes through the boys and looks for a specific variable n, which has been assigned a string.  

#after searching through all it prints a statement with the the boys with n as a name

        

for name in girls.keys():

    if name == n:

        print("There were "+ str(girls[name]) + " girls named " + n)



#for loop that goes through the girls and looks for a specific variable n, which has been assigned a string.  

#after searching through all it prints a statement with the the girls with n as a name