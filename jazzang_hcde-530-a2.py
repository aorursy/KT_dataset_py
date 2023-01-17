mylist = [1, 2, 3, 4, 5, 6]



# Define a variable i to iterate over each element of mylist

for i in mylist:

    # Printing each element on a new line 

    print(i)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

####



# Define accumulator variable total

total = 0

# Iterate through the items in the sequence

for i in mylist:

    # Update the accumulator variable each time 

    total = total + i 

# Print final accumulator variable result

print(total)
s = "This is a test string for HCDE 530"

# Add your code below

###

# Creates a list of separate words separated by a whitespace 

words = s.split()

# Iterate through the items in the list & print each word one by one

for i in words:

    print(i)
# Using enumerate method to find and replace the element with the value four 

for n, i in enumerate(mylist):

    if i == 4:

        mylist[n] = 'four'

print (mylist)



# Using indexing method. I commented this method just in case it affects other parts of the document 

# mylist[3] = 'four'

# print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

# Read and print each line in the text file 

for line in fname:

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



# Iterate over dogList

for dog in dogList: 

    # if found 'terrier', the character number when the string starts goes to the result

    result = dog.find('terrier')

    # if the result comes back negative, find 'Terrier'

    if result == -1:

        result = dog.find('Terrier')

    # print the dog name and the char number for easier display 

    print(dog, result)
binList = [0,1,1,0,1,1,0]



# Add your code below

# Define variable i to iterate over binList

for i in binList:

    # If current item equals 1 print "One"

    if i == 1:

        print('One')

        

    ### To print a space if i != 1, but the question asked to don't print anything hence I commented this section

    #else:

        #print(' ')
# Define a variable word to input the string to detect and print 

word = "Bulldog"



### Method 1: Using conditionals with find & lower method 

for dog in dogList: 

    if dog.lower().find(word.lower()) != -1:

        print(dog)



### Method 2: Using conditionals with lower method only 

# Iterate over dogList and then use if and lower method 

# for dog in dogList:

#     if word.lower() in dog.lower():

#         print (dog)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    

    # Split the string into sub strings delimited by whitespace and then count the number of the words

    words = line.split()

    numWords += len(words)

    

    # Strip newline characters to EXCLUDE newline characters in character count.

    line = line.rstrip()

    # Also written as numChars = numChars + 1

    numChars += len(line)

    

    # Also written as numLines = numLines + 1

    numLines += 1



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)

# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Define variables for the number of characters, lines and words

sherlockChars = 0

sherlockLines = 0

sherlockWords = 0



# Create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



# Read each line in the file, count the number of characters, lines, and words

# Updating the numChars, numLines, and numWords variables.

for line in fname:

    

    # Split the string into sub strings delimited by whitespace and then count the number of the words

    words = line.split()

    sherlockWords += len(words)

    

    # Strip newline characters to EXCLUDE newline characters in character count

    line = line.rstrip()

    # Also written as sherlockChars = sherlockChars + 1

    sherlockChars += len(line)

    

    # Also written as sherlockLines = sherlockLines + 1

    sherlockLines += 1



# Print the respective variables after tabulation

print('%d characters'%sherlockChars)

print('%d lines'%sherlockLines)

print('%d words'%sherlockWords)



# Close your file

fname.close()