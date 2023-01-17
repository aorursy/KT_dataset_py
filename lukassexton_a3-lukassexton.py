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



NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"

boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls

alphabet = {'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0, 'g':0, 'h':0, 'i':0, 'j':0, 'k':0, 'l':0, 'm':0, 'n':0, 'o':0, 'p':0, 'q':0, 'r':0, 's':0, 't':0, 'u':0, 'v':0, 'w':0, 'x':0, 'y':0, 'z':0} # empty dictionary that stores number of names that start with each letter



for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")

    count = int(count)   # Cast the string 'count' to an integer

   

    if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

        girls[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys[name.lower()] = count



        

# name is the key in dict:boys 

#First letter is taking the first letter of the string 

        

        

for temp_name in boys:

    first_letter = temp_name[0] # assign first letter of name to a variable

    #print(first_letter)

    # loop through alphabet

  #  for letter in alphabet.keys(): # Can it be for lettter in alphabet?

 #       if letter == first_letter:

  #          alphabet[letter] += 1

    for letter in alphabet: # Can it be for lettter in alphabet?

        if letter == first_letter:

            alphabet[letter] += 1

    

    

for temp_name in girls:

    first_letter = temp_name[0] # assign first letter of name to a variable

    for letter in alphabet: # Can it be for lettter in alphabet?

        if letter == first_letter:

            alphabet[letter] += 1

    #print(first_letter)

    # loop through alphabet

  #  for letter in alphabet.keys():

   #     if letter == first_letter:

    #        alphabet[letter] += 1





for letter in alphabet:

    print(str(alphabet[letter]) + " baby names begin with the letter %s" %letter)

       

#print("There were %d girls' names." %len(girls))

#print("There were %d boys' names." %len(boys))
type(count)
total_count = 0

    # Since 'count' is actaully a string of text and not an integer, 

    # we need to turn it into an integer to store that number in the dictionary so we can use it. 

    # later to do arithmetic that we couldn't do, if it was just text. This is called 'casting'.

    

for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    # Since 'count' is actaully a string of text and not an integer, 

    # we need to turn it into an integer to store that number in the dictionary so we can use it. 

    # later to do arithmetic that we couldn't do, if it was just text. This is called 'casting'.

    

    count = int(count)   # Cast the string 'count' to an integer

    total_count += count

   # print(total_count)

# print("There were %d girls' names." %int(total_count))

print ("There are %d babies in 2010 dataset" %int(total_count))

#print(total_count)





#boys(john) += 1





#    if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

#        girls[count] = count

#        # store the current girls name we are working with in the dictionary, with its count

#        print (int(girls[count]), str("Test F"))

#    elif gender == "M": # Otherwise store it in the boys dictionary

#        boys[count] = count

#        print (int(boys[count]), str("Test M"))





#for total_count in boys:

#    print (int(girls[count]))
longest_string_male = max(boys, key=len)

print("The longest boy name in the dataset is " + longest_string_male)



longest_string_female = max(girls, key=len)

print("The longest girl name in the dataset is " + longest_string_female)



m = int(len(longest_string_male))

f = int(len(longest_string_female))



if f > m:

    print("The longest name in the dataset is ", longest_string_female)

elif f < m:

    print ('The longest name in the dataset is ' + longest_string_male)

else:

    print("Both " + longest_string_female + " and " + longest_string_male + "are equal in length")
longest_string_male = max(boys, key=len)

print("The longest boy name in the dataset is " + longest_string_male)



longest_string_female = max(girls, key=len)

print("The longest girl name in the dataset is " + longest_string_female)



if longest_string_male > longest_string_female:

    print("The longest name in the dataset is " + longest_string_female)

elif longest_string_male < longest_string_female:

    print ('The longest name in the dataset is ' + longest_string_male)

else:

    print("Both " + longest_string_female + " and " + longest_string_male + "are equal in length")

    

    
longest_string_male = max(boys, key=len)

print("The longest boy name in the dataset is " + longest_string_male)



longest_string_female = max(girls, key=len)

print("The longest girl name in the dataset is " + longest_string_female)



if longest_string_female > longest_string_male:

    print("The longest name in the dataset is " + longest_string_female)

elif longest_string_female < longest_string_male:

    print ('The longest name in the dataset is ' + longest_string_male)

else:

    print("Both " + longest_string_female + " and " + longest_string_male + "are equal in length")
n_count = 0

gn_count = 0

NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"

boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls



#For Loop creates boy and girls dictionaries 

for line in open(NAMES_LIST, 'r').readlines():

    name, gender, count = line.strip().split(",")

    count = int(count)   # Cast the string 'count' to an integer

   

    if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

        girls[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys[name.lower()] = count

        

for gender_boy in boys: # First boy name is called Gender boy 

    exist = girls.get(gender_boy,0)

    if exist > 0:

        gn_count += 1





        

for gender_boy in boys: # First boy name is called Gender boy 

    exist = girls.get(gender_boy,1)

    if exist > 0:

        n_count += 1        

        

    #for gender_girl in girls:  # First girl name is called Gender girl 

    #   if gender_boy == gender_girl: # Are the names equivalent?

    #        gn_count += 1 # 

#   #         print(gn_count)

#    print(gn_count)

#print(gn_count)

# dict.get([key,default]) 







print("There were %d gender neutral names." %(gn_count))

print("There were %d total names." %(n_count))
# Comp Asks the user to input a name to search the 2010 baby database for comparison

print ('Please enter a name (CASE SENSITIVE) to return the number of babies (M & F) with similar names as the input from 2010')



# Simultansly asks the user for their input while defining the variable they submit at user_input

user_input = input()



#Python is printing the input automatically, so this attempts at displaying the input within the context of the query

#print ('User queried ' + (user_input))



for name in boys.keys():

    if user_input in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if user_input in name:

        print(name + " " + str(girls[name]))



print (user_input)
# Comp Asks the user to input a name to search the 2010 baby database for comparison

print ('Please enter a name (CASE SENSITIVE) to return the number of babies (M & F) with similar names as the input from 2010')



# Simultansly asks the user for their input while defining the variable they submit at user_input

user_input = input()



#Python is printing the input automatically, so this attempts at displaying the input within the context of the query

#print ('User queried ' + (user_input))



for name in boys.keys():

    if user_input == name:

        print("There are these many " + name + " boys " + str(boys[name]))



for name in girls.keys():

    if user_input == name:

        print("There are these many " + name + " girls " + str(girls[name]))



#print (%d girls named user_input and there are these many boys named ryan)
#NAMES_MASTER = 

NAMES_LIST_2010 = "/kaggle/input/a3data/yob2010.txt"

NAMES_LIST_2011 = "/kaggle/input/a3data/yob2011.txt"

NAMES_LIST_2012 = "/kaggle/input/a3data/yob2012.txt"

NAMES_LIST_2013 = "/kaggle/input/a3data/yob2013.txt"



boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls





#def fileReader(filename)

#    process each line in filename

#    filepath =  "/kaggle/input/a3data/" + filename

#    for line in open(filepath, 'r').readlines():

#    accumulate the results and 

#    return the result





mylist = ['yob2010.txt', 'yob2010.txt', ...]



#process all the lists

#for item in mylist:

#    fileReader(item)

#    process the data

#    add up the numbers

    

#for (int i=0;item<len(mylist);i++){

#    do this to (item[i])

#    next i;

#}

    

#2010

for line in open(NAMES_LIST_2010, 'r').readlines():  # iterate through the lines and separate each line on its commas

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



print("In 2010, there were %d girls' names." %len(girls))

print("In 2010, there were %d boys' names." %len(boys))



#2011



for line in open(NAMES_LIST_2011, 'r').readlines():  # iterate through the lines and separate each line on its commas

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



print("In 2011, there were %d girls' names." %len(girls))

print("In 2011, there were %d boys' names." %len(boys))



#2012

for line in open(NAMES_LIST_2012,'r').readlines():  # iterate through the lines and separate each line on its commas

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



print("In 2012, there were %d girls' names." %len(girls))

print("In 2012, there were %d boys' names." %len(boys))



#2013

for line in open(NAMES_LIST_2013, 'r').readlines():  # iterate through the lines and separate each line on its commas

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



print("In 2013, there were %d girls' names." %len(girls))

print("In 2013, there were %d boys' names." %len(boys))







# We did this for you at the end of the previous homework.

# It's a little weird to stuff numbers into sentences this way, but once you get 

# used to it, it's easy. You can do lots of other formatting like this.

# Here's an explanation of how it works: https://www.geeksforgeeks.org/python-output-formatting/