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



#Take input and look up the count of that name

name2 = input()

print("There were " + str(boys[name2]) + " boys named " + name2)



for name in boys.keys():

    if name in girls.keys():

        print(name)
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))



# 1(a) Are there more boy names or girl names?



#Compare the number of girl and boy names in the dataset and print message accordingly

if len(boys)>len(girls):

    print("There are more boy names in the dataset."+" -- boys: "+str(len(boys)) +" girls: "+ str(len(girls)))

elif len(boys)<len(girls):

    print("There are more girl names in the dataset."+" -- boys: "+str(len(boys)) +" girls: "+ str(len(girls)))

else:

    print("There are an equal number of girls and boys names in the dataset."+" -- boys: "+str(b_count) +" girls: "+ str(g_count))
#Ask user for input

print("Type a single letter to see whether there are more boy or girl names starting with that letter: ")

letter = input()[0]; #take the first character of the input



# Takes letter as input

# returns count of girls names beginning with that letter.

def count_girl_names_by_letter(letter):   

    numGirlsNames = 0 #create counter for girls names

    #Iterate over girls names

    #If the name starts with the letter, increment numGirlsNames

    for g in girls:

        if g[0] == letter:

            numGirlsNames += 1

    return numGirlsNames



# Takes letter as input

# returns count of boys names beginning with that letter.

def count_boy_names_by_letter(letter):

    numBoysNames = 0 #create counter for boys names

    #Iterate over boys names

    #If the name starts with the letter, increment numBoysNames

    for b in boys:

        if b[0] == letter:

            numBoysNames += 1

    return numBoysNames





b_count = count_boy_names_by_letter(letter) #Count boys names that begin with a particular letter

g_count = count_girl_names_by_letter(letter) #Count girls names that begin with a particular letter



#If no names begin with that character (say, a number input), print message saying so

if b_count==0 and g_count ==0:

    print("No names in the dataset start with: " + letter)



#Compare b_count and g_count and print message saying

# whether there are more girls names, more boys names, or an equal amount of both

elif b_count > g_count:

    print("There are more boys names starting with the letter "+letter+ " (boys: "+str(b_count) +" girls: "+ str(g_count)+")")

elif b_count < g_count:

    print("There are more girls names starting with the letter "+letter+ " (boys: "+str(b_count) +" girls: "+ str(g_count)+")")

elif b_count == g_count != 0:

    print("There are an equal number of girls and boys names in the dataset starting with "+letter+ " (boys: "+str(b_count) +" girls: "+ str(g_count)+")")

    



# 1(c) Are there more boy names or girl names for every letter?

#



# import string library function  

import string



#List of every letter in the alphabet

alphabet = list(string.ascii_lowercase)



#For each letter of the alphabet

for l in alphabet:

    #Count the number of boy and girl names for that letter

    b_count= count_boy_names_by_letter(l)

    g_count= count_girl_names_by_letter(l)

    

    #compare the counts and pick "winning" gender for that letter

    winning_gender=""

    if b_count> g_count:

        winning_gender="boys"

    elif b_count< g_count:

        winning_gender="girls"



    #print out the letter, totals

    print(l+": More "+winning_gender+" names \t boys: "+str(b_count)+"\t girls: "+str(g_count))
#set up counter for babies

total_babies = 0



#iterate over every item in boys and the count to the total

for b in boys.keys():

    total_babies += boys[b]



#iterate over every item in girls and the count to the total

for g in girls.keys():

    total_babies += girls[g]



print("There are "+str(total_babies)+" babies in the dataset.")
#set up list to hold the longest names found so far

longest_names=[""]



#merge girl and boy names into a signle list

all_names = []

for g in girls.keys():

    all_names.append(g)

for b in boys.keys():

    all_names.append(b)



#iterate over all names

for i in all_names:

    #if name is longer than current longest_name, replace longest_name with it

    if len(i) > len(longest_names[0]):

        longest_names.clear()

        longest_names.append(i)

    #if name is the same length than current longest_name, add the name to the list

    elif len(i) == len(longest_names[0]):

        longest_names.append(i)



print("The longest name(s) in the dataset are:")

for i in longest_names:

    print(i)



print("They are "+str(len(longest_names[0]))+" characters long")
#set up counter of gender neutral names

gender_neutral=0



#for each name in girls, check if it also exists in boys

for name in girls.keys():

    if name in boys.keys():

        gender_neutral +=1 #if so, increment gender_netural by 1



print("There are "+str(gender_neutral)+ " names in both the girls and boys datasets.")
#set up a list for names which are subsets of other names

names_that_are_substrings=[""]



#get rid of duplicates in all_names

all_names_unique = list(dict.fromkeys(all_names))

#sort names alphabetically (made it easier for me when I was testing things out)

all_names_unique = sorted(all_names_unique)



#look through each name in all_names_unique

#Note: this takes a long time. Would like to find a faster method.

for name in all_names_unique:

    #compare each name against every other name

    for name2 in all_names_unique:

        #if the name is in another name, but they are not the same

        if name in name2 and name != name2:

            names_that_are_substrings.append(name) #add it to the list

            print(name+" is a subset of "+name2)

            break #stop looking



print("There are "+ str(len(names_that_are_substrings))+" names that are substrings of other names.")
#import collections so that we can use an ordered dictionary

import collections



#sort girls names by popularity and store them in an ordered dictionary

girls_by_popularity = collections.OrderedDict(girls,reverse = True)



#starting with most popular girls name, check if it is in the boys list

for name in girls_by_popularity:

    #if it is, print output and stop searching

    if name in boys: 

        print("The most popular girls name that is also a boys name is: "+name)

        break
print("Enter a name to see how many girls and boys were born with that name: ")



#take input

name = input()



if name in girls and name in boys:

    print("There are %i girls and %i boys with the name %s " %(girls[name], boys[name], name))

elif name in girls:

    print("There are %i girls and 0 boys with the name %s " %(girls[name], name))

elif name in boys:

    print("There are %i boys and 0 girls with the name %s " %(boys[name], name))

else:

    print("No matches found for "+name)
print("Enter a prefix to see which girls and boys names start with that prefix: ")



#take input

prefix = input()



#counters for number of babies with that prefix in each dataset

girls_with_prefix=0

boys_with_prefix=0



#iterate through girls dataset

for g in girls:

    # if the name starts with the prefix

    if g.find(prefix) == 0:

        #add that name's count to girls_with_prefix

        girls_with_prefix += girls[g]



#iterate through boys dataset

for b in boys:

    # if the name starts with the prefix

    if b.find(prefix) == 0:

        #add that name's count to boys_with_prefix

        boys_with_prefix += boys[b]



print("There are "+str(girls_with_prefix)+" girls and "+str(boys_with_prefix)+" boys with names starting with \""+ prefix+"\"")

NAMES_LIST_ALL_YEARS= ["/kaggle/input/a3data/yob2010.txt", "/kaggle/input/a3data/yob2011.txt", "/kaggle/input/a3data/yob2012.txt", "/kaggle/input/a3data/yob2013.txt"]



boys_all_years = {}  # create an empty dictionary of key:value pairs for the boys

girls_all_years = {} # create an empty dictionary of key:value pairs for the girls



#Build dictionaries of girls and boys names from all 4 years

for file in NAMES_LIST_ALL_YEARS:

                          

    for line in open(file, 'r').readlines():  # iterate through the lines and separate each line on its commas

        # since we know there are three items on each line, we can assign each of them to a variable

        # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

        name, gender, count = line.strip().split(",")



        count = int(count)   # Cast the string 'count' to an integer

         

        # If it's a girl, save it to the girls dictionary, with its count

        # If already exists in the dictionary, update the count of the name

        if gender == "F":    

            if name.lower() in girls_all_years: #if the name is already in the dictionary

                girls_all_years[name.lower()] += count #update count of the name

            else:

                girls_all_years[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

        

        elif gender == "M": # Otherwise store it in the boys dictionary

            if name.lower() in boys_all_years: #if the name is already in the dictionary

                boys_all_years[name.lower()] += count #update count of the name

            else:

                boys_all_years[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    



#Find most popular boys name across all 4 years

most_popular_boys_name = ""

most_popular_boys_name_count =0



#iterate over all boys names and update the highest count found

for b in boys_all_years:

    if boys_all_years[b] >most_popular_boys_name_count:

        most_popular_boys_name = b

        most_popular_boys_name_count = boys_all_years[b]



print("The most popular boys name is "+most_popular_boys_name+" : "+ str(most_popular_boys_name_count))



#Find most popular girls name across all 4 years

most_popular_girls_name = ""

most_popular_girls_name_count =0



#iterate over all girls names and update the highest count found

for g in girls_all_years:

    if girls_all_years[g] >most_popular_girls_name_count:

        most_popular_girls_name = g

        most_popular_girls_name_count = girls_all_years[g]



print("The most popular girls name is "+most_popular_girls_name+" : "+ str(most_popular_girls_name_count))

NAMES_LIST_2010= "/kaggle/input/a3data/yob2010.txt"

NAMES_LIST_2013= "/kaggle/input/a3data/yob2013.txt"





boys_2010 = {}  # create an empty dictionary of key:value pairs for the boys in 2010

girls_2010 = {} # create an empty dictionary of key:value pairs for the girls in 2010

boys_2013 = {}  # create an empty dictionary of key:value pairs for the boys in 2013

girls_2013 = {} # create an empty dictionary of key:value pairs for the girls in 2013



#Build dictionary of girls and boys names For 2010

for line in open(NAMES_LIST_2010, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    count = int(count)   # Cast the string 'count' to an integer

         

    # If it's a girl, save it to the girls dictionary, with its count

    # If already exists in the dictionary, update the count of the name

    if gender == "F":    

        girls_2010[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count        

    elif gender == "M": # Otherwise store it in the boys dictionary

       boys_2010[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count



#Build dictionary of girls and boys names For 2013

for line in open(NAMES_LIST_2013, 'r').readlines():  # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    count = int(count)   # Cast the string 'count' to an integer

         

    # If it's a girl, save it to the girls dictionary, with its count

    # If already exists in the dictionary, update the count of the name

    if gender == "F":    

        girls_2013[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count        

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys_2013[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    



    

#Compare boys and girls names in 2010 vs 2013

girls_diff= {} #create empty dictionary to hold differences in girl name popularities

boys_diff={} #create empty dictionary to hold differences in boy name popularities



#Takes two dictionaries as input

#Finds differences between values for keys that are in both dictionaries them

#Returns dictionary with the differences

def diff_names(early_dataset, later_dataset):   

    diffs={} # define dictionary to store differences   

    for name in later_dataset:

        if name in early_dataset:

            diffs[name] = later_dataset[name] - early_dataset[name] #store the difference

            #print statement used to debug differences, comment out when done

            #print("Name:"+name+" \t 2013: "+str(early_dataset[name])+" \t 2010:"+str(later_dataset[name])+" \t Diff:"+str(diffs[name]))

    return diffs



girls_diff= diff_names(girls_2010, girls_2013)

boys_diff= diff_names(boys_2010, boys_2013)





#Find the girls and boys names with the highest rise in popularity (biggest difference)

boy_biggest_diff = 0 #holds change from 2010 to 2013

boy_biggest_diff_name = "" #holds boys name



girl_biggest_diff = 0 #holds change from 2010 to 2013

girl_biggest_diff_name = "" #holds boys name



#Find the girls name with the biggest change

for g in girls_diff:

    if girls_diff[g] > girl_biggest_diff:

        girl_biggest_diff = girls_diff[g]

        girl_biggest_diff_name = g



#Find the boys name with the biggest change

for b in boys_diff:

    if boys_diff[b] > boy_biggest_diff:

        boy_biggest_diff = boys_diff[b]

        boy_biggest_diff_name = b    

        

print("The girls' name with the biggest increase in popularity between 2010 and 2013 is %s. It increased by %i." %(girl_biggest_diff_name, girl_biggest_diff))

print("The boys' name with the biggest increase in popularity between 2010 and 2013 is %s. It increased by %i." %(boy_biggest_diff_name, boy_biggest_diff))

        


