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

#1. Are there more boy names or girl names? What about for particular first letters? What about for every first letter?



#I did an if/elif statement to take the result of exercise 1 and printed the statement of more boys or girls based on the results of len(girls).



if len(girls) > len(boys):

    print("there were more girls")

elif len(girls) < len(boys):

    print("there were more boys")



#Then I created a new dictionary called Alphabet and counted names/letters within the dictionary and printed out the count. I really struggled with this. I think the other way to do this is to list all unique position 0 values of name and then count them.

    

alphabet = {}

for name in boys.keys():

    name = name.strip()

    starts_with = name[0]

    if starts_with in alphabet:

        alphabet[starts_with]+= 1

    else:

        alphabet[starts_with] = 1

print(alphabet)



#2. How many babies are in the dataset (assuming nobody is counted more than once)?



#I assumed that the "babies" in the dataset consisted of everyone in the dataset since the title was Year of Birth 2010 and that I would not have to filter by a separate date of birth.

#As a result, I assumed that the # of lines in the dataset would be equivalent to the # of the babies.

#I basically just used a readlines function on the original NAMES_LIST file and then counted the number of lines with len. Then I converted into an integer, followed by a string to be able to print.



bcount = len(open(NAMES_LIST).readlines(  ))

bcount = int(bcount)  

print("# of babies "+ str(bcount))



#4. How many boy names are also girl names? How many girls' names are also boys' names?



#I created a new dictionary called alsoboys. Then I used get to +1 every time a name in boys shows up in girls and I used the len function to fill in the sentence and print the dictinoary.

alsoboys = {}

for name in boys.keys():

    if name in girls.keys():

        alsoboys[name] = alsoboys.get(name, 0) + 1

print("there were %d shared boys names with girls names"%len(alsoboys))



#for girls, I created the same process but in reverse.

alsogirls = {}

for name in girls.keys():

    if name in boys.keys():

        alsogirls[name] = alsogirls.get(name, 0) + 1

print("there were %d shared girls names with boys names"%len(alsogirls))
