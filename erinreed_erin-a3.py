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

#Exercise 1A: Are there more boy names or girl names? 
#to find out if there are more girls or boys name I compared the number of girls names to boys names:

if len(girls) > len(boys):
    print("There are more girls names than boys names")

if len(girls) < len(boys):
    print("There are less girls names than boys names")

if len(girls) == len(boys):
    print("There are the same number of girls names as boys names")





 


     

#Exercise 1B: What about for particular first letters?
#Using the letter "b" as a letter example I will total the number of boys compared to the total number of girls names that start with the letter "b" in the 0 index position.

boys_numlines=0
girls_numlines=0

#looking in each dictionary for names that start with the letter "b" and then adding the total lines.
for name in boys.keys():
    if "b" in name[0]:
        boys_numlines += 1
        
for name in girls.keys():
    if "b" in name[0]:
        girls_numlines += 1

#since each girl and boy name that start with "b" are given on a seperate line, the comparison between total number of lines for each will tell me if there are more girls names that start with the letter "b."       
print("For names that start with b there are more girl names than boy names: " + str(girls_numlines > boys_numlines)) 

#This code will give us additional information that there is in fact the same number of names in each dictionary for the particular.
if girls_numlines == boys_numlines:
    print("There are the same number of boys names as girls names that start with b")

#Exercise 1C: What about for every first letter?
#using a variable to define the first letter of the name, so you can check for all first letters.
#Using the same equation from 1B, but now defining the first letter as a variable that you can easily adjust.
boys_numLines=0
girls_numLines=0
letter="j"

for name in boys.keys():
    if letter in name[0]:
        boys_numLines += 1
        
for name in girls.keys():
    if letter in name[0]:
        girls_numLines += 1

#printing string statement that describes what the output is telling us when "letter" is defined. 
if girls_numLines > boys_numLines:
    print("When the first letter is " + letter + " there are more girls names than boys names")
if girls_numLines < boys_numLines:
    print("When the first letter is " + letter + " there are more boys names than girls names")
if girls_numLines == boys_numLines:
    print("When the first letter is " + letter + " there are the same number of boys names as girls names")
    
    
#Exercise 2: How many babies are in the dataset (assuming nobody is counted more than once)?

#finding how many values are associated with each key in the boy name dictionary
values_boys = boys.values()


#finding how many values are associated with each key in the girl name dictionary
values_girls = girls.values()

   
#adding the total values for girls and boys names    
print("Total number of babies in dataset= " + str(sum(values_girls)+sum(values_boys)))


#Exercise 3: What is the longest name in the dataset?
#I used a little trial and error to narrow down to the names with the lonest length. There were no names over the length of 15.

#Here I have printed the 3 longest boy names (each with 15 characters)
print("The longest boy names")
for name in boys.keys():
    if len(name)>14:
        print(name)
        

#Here I have printed the 3 longest girl names (each with 14)
print("\n" + "The longest girl names")
for name in girls.keys():
    if len(name)>13:
        print(name)
        

        
#Exercise 4: How many boy names are also girl names? How many girls' names are also boys' names?

boys_numLines=0
girls_numLines=0

#this code is finding for any names that are in the boys dictionary that are also in the girls dictionary and then printing the total number of names in the at list.
for name in boys.keys():
    if name in girls.keys():
        boys_numLines += 1
print(boys_numLines)



#this code starts with all names in the girl dictionary and looks for those names in the boys dictionary.
#this code produces the same results as the code above as we are looking for names that live in both dictionaries so it doesn't matter if the name was found in the girl dictionary vs. the boy dictionary first.
for name in girls.keys():
    if name in boys.keys():
       girls_numLines += 1
print(girls_numLines)    
#Exercise 5: How many names are subsets of other names?
#I wasn't able to complete this question but here is my thought process for starting the process to search for a name that is subset.

#I found how to count how many times a single name ex. "jacob" appears as a name or subset of other names in the boy's dictionary.

boysSubset_numLines=0
search_name = "jacob"

for name in boys.keys():
    if search_name in name:
        boysSubset_numLines += 1   
print(boysSubset_numLines)
    

# showing names where "jacob" exists in other names
for name in boys.keys():
    if search_name in name:  
            print(name)

            
    
     
    

#Exercise 6: What is the most popular girl name that is also a boy name?


#finding the frequecy of a name occuring in the girls dictionary and iterarting through the value for each key. 
#finding the girl name that has the greatest value
max_count = 0
for name in girls:
#making sure the name is also a key in the boys dictionary
    if name in boys:
        if girls[name] > max_count:
#assigning the value for that key and the name to a variable.
            max_count = girls[name]
            maxword = name
print(f"{maxword} is the most popular girl name that is also a boy name, It appears in the girls dictionary {max_count} times.")

    

  



#Exercise 7: Write a program that will take a name as input and return the number of babies with that name in the girl and boy datasets.

#input name as a variable 
inputname = "joanna"

#using the .get function to print the value for that name in both the girls and boys dictionary and return a default value if the name is not found.
print(girls.get(inputname, "There are no girls with this name"))
print(boys.get(inputname, "There are no boys with this name")) 






#Exercise 8: Take a prefix as input and print the number of babies with that prefix in each dataset (i.e., "m" would list babies whose names start with "m" and "ma" would list babies whose names start with "ma", etc).
#assigning variables 
pre_numBoys=0
pre_numGirls=0
input_prefix = "m"

#looking in the boys dictionary for names that start with "m" in the first two position [0]
#running a total 
for name in boys.keys():
    if input_prefix in name[0]:
        pre_numBoys += 1   
    
#looking in the girls dictionary for names that start with "m" in the first two position [0]
#running a total 
for name in girls.keys():
    if input_prefix in name[0]:
        pre_numGirls += 1   
        
#assigning a variable to the total number that includes boys and girls with this prefix.
total = (pre_numBoys + pre_numGirls)
   
print("Total number of babies in dataset with m as prefix = " + str(total))



#assigning new variable to find a longer prefix and provide new totals   
pre_numBoystwo=0
pre_numGirlstwo=0
input_prefixtwo = "ma"

#looking in the boys dictionary for names that start with "ma" in the first two position [0:2]
#running a total 
for name in boys.keys():
    if input_prefixtwo in name[0:2]:
        pre_numBoystwo += 1   
    
#looking in the girls dictionary for names that start with "ma" in the first two position [0:2]
#running a total 
for name in girls.keys():
    if input_prefixtwo in name[0:2]:
        pre_numGirlstwo += 1   
        
#assigning a variable to the total number that includes boys and girls with this prefix.
total_two = (pre_numBoystwo + pre_numGirlstwo)
   
print("Total number of babies in dataset with ma as prefix = " + str(total_two))