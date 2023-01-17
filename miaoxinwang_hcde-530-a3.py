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

# commented for less output



# for name in boys.keys():

#     if name in girls.keys():

#         print(name)
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))

# 1. Are there more boy names or girl names? What about for particular first letters? What about for every first letter?

import string

# get a to z

alphabets = string.ascii_lowercase



print(f"initial: {len(boys)}\n")



# I guess I've got the question wrong initially?... The following is the solution to compare the # of babies by gender...

print("Question 1: - Compare the # of babies by gender...\n")



# For general name count

general_name_count = { 'boys': 0, 'girls': 0 }



# For every first letter

alphabet_name_count_boy = {}

alphabet_name_count_girl = {}



# set default count for the dictionary 

for i in alphabets:

    alphabet_name_count_boy[i] = 0

    alphabet_name_count_girl[i] = 0



# iterate through name sets and add up count by first letter in name

def get_name_count_by_alphabet(gender):

    if gender == 'boys':

        alphabet_name_count = alphabet_name_count_boy

        nameset = boys

    elif gender == 'girls':

        alphabet_name_count = alphabet_name_count_girl

        nameset = girls

        

    for name in nameset:

        count = nameset[name]

        # update overall name count

        general_name_count[gender] += count

        if len(name):

            first_letter = name[0]

            # update name count for the specific first letter

            alphabet_name_count[first_letter] += count



# General conclusion printing method

def name_count_comparison(boy_count, girl_count):

    if boy_count < girl_count:

        return 'more girls name'

    elif boy_count > girl_count:

        return 'more boys name'

    else:

        return 'same quantity for girls and boys names'



get_name_count_by_alphabet('boys')

get_name_count_by_alphabet('girls')



# Print out result for general name count

print(f"In general, there are {name_count_comparison(general_name_count['boys'], general_name_count['girls'])}\n")



# Print out result for each alphabet

for alphabet in alphabet_name_count_boy:

    print(f"For alphabet {alphabet}, there are {name_count_comparison(alphabet_name_count_boy[alphabet], alphabet_name_count_girl[alphabet])}")

    

    



# 2nd solution if the question is asking to compare the # of unique names by gender...

print("\n\nQuestion 1: - Compare the # of unique names by gender...")

# Print out result for general name count comparison

print(f"In general, there are {name_count_comparison(len(boys), len(girls))}\n")



# Print out result for each alphabet

result = {}

for i in alphabets:

    result[i] = 0



for boy_name in boys:

    result[boy_name[0]] += 1



for girl_name in girls:

    result[girl_name[0]] -= 1

        

for letter in result:    

    if result[letter] > 0:

        print(f"For alphabet {letter}, there are more boy names")

    elif result[letter] < 0:

        print(f"For alphabet {letter}, there are more girl names")

    else:

        print(f"For alphabet {letter}, # of boy and girl names are the same")



    

    


# 2. How many babies are in the dataset (assuming nobody is counted more than once)?

print("\nQuestion 2:")

print(f"In total, there are {general_name_count['boys'] + general_name_count['girls']} new baby counts.")


# 3. What is the longest name in the dataset?

print("\nQuestion 3:")



# default dict for storing result

longest_name_item = {'length': 0, 'sets': {''}}



for key in boys:

    if len(key) > longest_name_item['length']:

        # clear current result set when find longer name

        longest_name_item['sets'].clear()

        longest_name_item['sets'].add(key)

        longest_name_item['length'] = len(key)

    elif len(key) == longest_name_item['length']:

        # keep adding name to the set

        longest_name_item['sets'].add(key)



print(f"The longest name in the dataset: {longest_name_item['sets']}")



# 4. How many boy names are also girl names? How many girls' names are also boys' names?

print("\nQuestion 4:")

# convert the key lists of the dict to sets and then get the intersect of them

boy_name_set = set(boys.keys())

girl_name_set = set(girls.keys())

intersection = boy_name_set.intersection(girl_name_set)

print(f"There are {len(intersection)} unisex names in the dataset.")



# 5. How many names are subsets of other names?

import time

start_time = time.time()



union_of_names = boy_name_set.union(girl_name_set)

count = 0

name_by_length = {}



for name in union_of_names:

    length = len(name)

    name_by_length.setdefault(length, set()).add(name)

    

name_set_list = sorted(name_by_length.items(), key=lambda t: t[0])

for index in range(len(name_set_list)):  

    for name in name_set_list[index][1]:

        j = index + 1

        found_match = False

        while j < len(name_set_list) and not found_match:

            for longer_name in name_set_list[j][1]:

                if name in longer_name:

                    count += 1

                    found_match = True

                    break

            j += 1



print(f"there are {count} names subsets of other names")

print("--- %s seconds ---" % (time.time() - start_time))


# 6. What is the most popular girl name that is also a boy name?

print("\nQuestion 6:")



# changing the original implementation to the function for the future usgae in question 9

def get_most_popular_name(name_set, name_count_dict):

    most_popular_name = {'count':0, 'sets': {''}}



    for name in name_set:

        if name_count_dict[name] > most_popular_name['count']:

            # if find more popular name, clear the sets and save the current one

            most_popular_name['sets'].clear()

            most_popular_name['sets'].add(name)

            most_popular_name['count'] = name_count_dict[name]

        elif name_count_dict[name] == most_popular_name['count']:

            # if multiple names have the same count, then keep adding into the sets

            most_popular_name['sets'].add(name)

    

    return most_popular_name



most_popular_name = get_most_popular_name(intersection, girls)

print(f"The most popular unisex name: {most_popular_name['sets']} with count {most_popular_name['count']}")

 
       

# 7. Write a program that will take a name as input and return the number of babies with that name in the girl and boy datasets.

print("\nQuestion 7:")



def get_baby_count_by_name(name):

    return boys.get(name, 0) + girls.get(name, 0)



baby_name = input('Type a baby name to see how many babies have the same name: ')

print(get_baby_count_by_name(baby_name))
# 8. Take a prefix as input and print the number of babies with that prefix in each dataset (i.e., "m" would list babies whose names start with "m" and "ma" would list babies whose names start with "ma", etc).

print("\nQuestion 8:")



name_prefix = input('Let us see what other options for the same prefix: ')

count_of_babies = 0



for name in boy_name_set.union(girl_name_set):

    if name.startswith(name_prefix):

        count_of_babies += 1



if count_of_babies:

    print(f"There are {count_of_babies} babies have the same prefix")

else:

    print(f"There is no baby having the same prefix")
# ### Extra Challenges (Note, you will need to load all of the datasets in order to solve these!)

# 9. Which boy and girl names are the most popular across all four years in our dataset? (hint: to solve this challenge, you will need to edit how data is loaded in the first code cell to include all the source files.)



ALL_NAME_LISTS = {"/kaggle/input/a3data/yob2010.txt", "/kaggle/input/a3data/yob2011.txt", "/kaggle/input/a3data/yob2012.txt", "/kaggle/input/a3data/yob2013.txt"}



all_babies = {}



for file_path in ALL_NAME_LISTS:

    # iterate through files

    current_file = open(file_path, 'r')

    

    for line in current_file.readlines():

        # iterate through lines and separate each line on its commas

        name, gender, count = line.strip().split(",")

        count = int(count)

        

        # if one name shows up in multiple datasets, the count should be added up

        all_babies[name.lower()] = all_babies.get(name.lower(), 0) + count 

            

    current_file.close()



names = get_most_popular_name(all_babies.keys(), all_babies)

print(f"{names['sets']} are the most popular names across all four years in the dataset with count {names['count']}")
# 10. Which boy and girl names have increased most in popularity between 2010 and 2013? Which ones have declined most in popularity? 



# Being lazy to refactor the above code, thus I will read the file again

babies_2013 = {}



current_file = open("/kaggle/input/a3data/yob2013.txt", 'r')



for line in current_file.readlines():

    # iterate through lines and separate each line on its commas

    name, gender, count = line.strip().split(",")

    count = int(count)



    # if one name shows up multiple times, the count should be added up

    babies_2013[name.lower()] = babies_2013.get(name.lower(), 0) + count 



current_file.close()



# get a union of all the babies name

names_across = set(babies_2013.keys()).union(boy_name_set)

diff = {}

increased_most = {'count': 0, 'sets': {''}}

decreased_most = {'count': 0, 'sets': {''}}



for name in names_across:

    # gets the difference in the count of name

    diff[name] = babies_2013.get(name, 0) - boys.get(name, 0) - girls.get(name, 0)

    

    # The following two blocks trying to get the most increaesd/descreased popularity name

    # The block could have been shortened... 

    if diff[name] > increased_most['count']:

        increased_most['sets'].clear()

        increased_most['count'] = diff[name]

        increased_most['sets'].add(name)

    elif diff[name] == increased_most['count']:

        increased_most['sets'].add(name)

    

    if diff[name] < decreased_most['count']:

        decreased_most['sets'].clear()

        decreased_most['count'] = diff[name]

        decreased_most['sets'].add(name)

    elif diff[name] == decreased_most['count']:

        decreased_most['sets'].add(name)

print(f"From 2010 to 2013, the following names increased the most: {increased_most['sets']} with count {increased_most['count']}")

print(f"From 2010 to 2013, the following names decreased the most: {decreased_most['sets']} with count {decreased_most['count']}" )