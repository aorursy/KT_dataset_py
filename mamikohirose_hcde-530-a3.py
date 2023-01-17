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

girls_names = 0 #create counter to start with 0 for both boys and girls

boys_names = 0



for name in girls:

    girls_names = girls_names +1 #add 1 to girls_name counter for every girl name and iterate through the girls keys

print("There are " + str(girls_names) + ' girl names.') #print total girl names



for name in boys: 

    boys_names = boys_names +1 #add 1 to boys_name counter for every boy name and iterate through the boys keys

print("There are " + str(boys_names) + ' boy names.') #print total boy names



if girls_names >= boys_names: 

    print("There are more girl names than boy names.")

else:

    print("there are more boy names than girl names")
girl_names = []#create empty lists

boy_names = []

print('Key in first letter of name you would like to search: ') #prompt for user to key in letter they want to search

first_letter = input() #input field



for name in girls: 

    if name[0] == first_letter: #for every name in girls, if the first letter is the same as what the user keyed in,

        girl_names.append([name])  #add that name to the empty list



print('there are ' + str(len(girl_names)) + " girl names that start with the letter " + first_letter + '.')  #print out message





for name in boys:

    if name[0] == first_letter:

        boy_names.append([name]) 



print('there are ' + str(len(boy_names)) + " boy names that start with the letter " + first_letter + '.') 
sum(boys.values())

print ('There are ' + str(sum(boys.values())) + ' baby boys in the dataset.')



sum(girls.values())

print ('There are ' + str(sum(girls.values())) + ' baby girls in the dataset.')



grandtotal = sum(boys.values())  + sum(girls.values())



print('There are ' + str(grandtotal) + ' total babies in the dataset')

longest_girl_name = "" #create empty variable

max_girl_length = 0 #counter starting with 0

longest_girlname_list = [] #create empty list



for name in girls: 

    if len(name) > max_girl_length: #if name is greater than max_length, add the name to the longest_name_list

        max_girl_length = len(name) #max_length becomes the iterated name

        longest_girl_name = name #also becomes the longest name

        longest_girlname_list =[] #empties out the longest_name_list

        longest_girlname_list.append(name) #appends the new longest name to the longest_name_list

    elif len(name) == max_girl_length: #if the length of the name is equal to the max_length, then also add the name to the list

        longest_girlname_list.append(name)

print('The longest girl name is ' + str(max_girl_length) + ' characters.')

print(longest_girlname_list)



longest_boy_name = "" #create empty variable

max_boy_length = 0 #counter starting with 0

longest_boyname_list = [] #create empty list



for name in boys: 

    if len(name) > max_boy_length: #if name is greater than max_length, add the name to the longest_name_list

        max_boy_length = len(name) #max_length becomes the iterated name

        longest_boy_name = name #also becomes the longest name

        longest_boyname_list =[] #empties out the longest_name_list

        longest_boyname_list.append(name) #appends the new longest name to the longest_name_list

    elif len(name) == max_boy_length: #if the length of the name is equal to the max_length, then also add the name to the list

        longest_boyname_list.append(name)

print('The longest boy name is ' + str(max_boy_length) + ' characters.')

print(longest_boyname_list)
boy_count = 0

girl_count = 0 



for name in boys.keys():

    if name in girls.keys():

        boy_count = boy_count + 1 

print ('There are ' + str(boy_count) + ' boy names that are also in girl names.')



for name in girls.keys():

    if name in boys.keys():

        girl_count = girl_count + 1

        

print ('There are ' + str(girl_count) + ' girl names that are also in boy names.')




