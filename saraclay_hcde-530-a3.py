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

# Comparing the number of boys names vs the number of girls names in the dictionaries

if ((len(boys)) > (len(girls))):

    print ("There are more boy names in total.")

else:

        print ("There are more girl names in total.")



# For each name in the boys dictionary

# Dictionaries have name, gender and count (in that order)

total_boys=0

total_girls=0



for name in boys.keys():

    # If a name starts with A

    if ((name[0]) == "a"): 

        total_boys=total_boys+1     



# For each name in the girls dictionary

for name in girls.keys():

    # If a name starts with A

    if ((name[0] == "a")):

        # total_girls += 1

        total_girls=total_girls+1



# Comparing the two totals from above

if ((total_girls) > (total_boys)):

    print ("There are more girls whose names start with a.")

else:

        print ("There are more boys whose names start with a.")

        

# How to do for every first letter?

# I want to iterate through an alphabet... but how do I do that in addition to already iterating through the separate dictionaries?

# Maybe try using the input method?



val = input("Enter a lowercase letter: ") 

for name in boys.keys():

    if ((name[0] == val)):

        total_boys=total_boys+1

for name in girls.keys():

        if ((name[0] == val)):

            total_girls = total_girls+1

        

if ((total_girls) > (total_boys)):

    print ("Girls: " + str(total_girls))

    print ("Boys: " + str(total_boys))

    print ("There are more girls whose names start with " + val)

elif ((total_girls==total_boys)):

    print ("There are equal number of girls and boys whose names start with " + val)

else:

     print ("There are more boys whose names start with " + val)
print("++++++++++++++++++++++++")

print("I'm showing my work here")

print("++++++++++++++++++++++++")

# I'm using the example from above to help me debug some of the code

total_queens=0

#total of this should just be 107 --> 70/26/6/5

for name in girls.keys():

    if 'queen' in name:

        number_queens = str(girls[name])

        print (number_queens)

        total_queens=number_queens + str(total_queens)

        print (total_queens)

        # when this prints out, everything is being appended to each other! how do we make this ADD and not APPEND?



total_queens2=0

for name in girls.keys():

    if 'queen' in name:

        number_queens2 = int(girls[name])

        total_queens2=number_queens2 + int(total_queens2)

        print (total_queens2)

        # when I change from "str" to "int" it added together correctly



print("++++++++++++++++++++++++")

# Here is where the actual code starts based on the work above

total_bbabies=0

# iterating through the loop

for name in boys.keys():

    number_name = int(boys[name[0:]])

    total_bbabies = number_name + (total_bbabies)

    

total_gbabies=0

for name in girls.keys():

    girl_number_name = int(girls[name[0:]])

    total_gbabies = girl_number_name + (total_gbabies)



    

print ("There are " + str(total_bbabies) + " boy babies.")

print ("There are " + str(total_gbabies) + " girl babies.")

print ("++++++++++++++++++++++++")

print ("Total number of babies: ")

print(total_bbabies+total_gbabies)

# I want to know how long each name in the boy name dataset is

# for name in boys.keys():

#        print (name) # also (name[0:])

#        print (len(name))

        

# For the actual code, I want to somehow compare the first name in the dictionary with the next name in the dictionary and so on forth.

# The first two names as fyi are jacob (5) and ethan (5)

# How might I do this? Let me see what I can do...



# for name in boys.keys():

#    x = (len(name[0:]))

    

#    if x > 10:

#        print ("I have a long name!")

        

# I know this is a cheap way of doing this, but there aren't that many names that have 10 characters are more.

# Now this will be more guessing. I know, this isn't the best of way of doing this, but my hope is that I can come up with a better function this way.



for name in boys.keys():

    x = (len(name[0:]))

    

    if x > 14:

        print ("The name " + name + " is a long name.")

    
# This is somewhat modified from the version above, so I'll start from there.

same_name=0

for name in boys.keys():

    if name in girls.keys():

        same_name = same_name+1



same_name2=0

for name in girls.keys():

    if name in boys.keys():

        same_name2 = same_name2+1



def samename():

    print ("There are " + str(same_name) + " girls that have the same names as boys.")

    print ("There are " + str(same_name2) + " boys that have the same names as girls.")

    

samename()

# Are they supposed to be the same total? This makes me question whether I got this right or not...
# This is about comparing keys within a dictionary to each other, which I don't know how to do (see #3).



# This gets all the names in the list "dict_keys". How can I access "dict_keys"?

# print (boys.keys())



# Ok, so now it looks like I turned this dictionary into a list

# print (list(boys))



# Now I can access the first name, yay!

# print (list(boys)[0])



# The code below spits out something interesting. The code is scanning the first name and seeing if there are other names in it.

# However, it's taking a long time because it's going through the ENTIRE list, and only for the first two names in the dictionaries.

# So I'm sort of close, but no cigar just yet.



for bname in boys.keys():

    b=0

    if (bname in (list(boys)[b])):

        print (bname)

        

for gname in girls.keys():

    g=0

    if (gname in (list(girls)[b])):

        print (gname)

        

# If I were to get this to work for every name, I would be able to fully answer this question. However, I'm not quite there yet. 
# For this, I'm going to start with the code that I created in #4 but modify it a little



for name in girls.keys():

    if name in boys.keys():

        print (name + " - " + str(girls[name]))

        

# Now that I'm more familiar with the data set, it seems like it's just printing out the female name data set in its entirety. Hmm...

# Anyway, it's like this in the example so I'll continue with it.

# Because I'm still stuck on comparing keys within the same dictionary, unfortunately I don't know how to continue here.



# First I want to branch out the two dictionaries and also offer an option if the input is invalid

# This looks ok to me... I'm not sure why it keep giving me the "'dict' object has no attribute 'key'" error...

gender_select = input("Would you like to look up a boy or girl name? Enter either boy or girl: ")

if (gender_select.lower() == "boy"):

    boy_choose = input ("Please enter a boy's name you'd like to look up: ")

    # 'dict' object has no attribute 'key'

    for bname in boys.key():

        if bname == boy_choose:

            print(bname)

        

elif (gender_select.lower() == "girl"):

    girl_choose = input ("Please enter a girl's name you'd like to look up: ")

    for gname in girls.key():

    # 'dict' object has no attribute 'key'

        if gname == girl_choose:

            print(gname)

    

else:

    print ("That is not a valid input.")
# I would probably need to solve for #7 first before I get to this problem...