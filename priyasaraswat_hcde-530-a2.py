mylist = [1, 2, 3, 4, 5, 6]

# Printing every item in the list using a variable each 

for each in mylist:

    print(each)

#Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0

# Value of every item is stored in variable each 

for each in mylist:

    # The number in mylist is added to the total using 'each' for every loop

    total = total + each

print(total)

s = "This is a test string for HCDE 530"

# A new string s1 saves the words in the string

s1 = s.split()

for each in s1:

    # Printing word per line

    print(each)
# Using function enumerate to store the index in 'n' and value in 'each'

for n, each in enumerate(mylist):

    # Using if to detect when the number is encountered

    if (each == 4):

        # Replacing '4' with a new value 'four'

        mylist[n] = 'four'

#Printing the updated list

print('The new list is', mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')

# Using rstrip the extra new line is eliminated from the output

for line in fname:

    line = line. rstrip('\n')

    print(line)

    

# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Iterating the list using 'i'

for i in dogList:

    # Using if and elif to find and print the index when 'terrier' or 'Terrier' is encountered

    if(i.find('terrier') != -1):

        print('Character number where the string terrier starts in',i,'is at', i.find('terrier'))

    elif(i.find('Terrier') != -1):

        print('Character number where the string Terrier starts in',i,'is at', i.find('Terrier'))

    else:

        # Printing '-1' in all other cases

        print('-1')

    
binList = [0,1,1,0,1,1,0]

#Iterating binList using for

for item in binList:

    # When if statement detects a '1' it'll print 'One' as output

    if(item == 1):

        print('One')

    else:

        continue
#Iterating doglist using for

for i in dogList:

    # Printing the breed that either has a 'Bulldog' or 'bulldog'

    if(i.find('Bulldog') != -1):

        print(i)

    elif(i.find('bulldog') != -1):

        print(i)

    else:

        continue
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for line in fname:

    # Using rstrip to identify the lines

    lines = line.rstrip('\n')

    # Using split to separate the words in a line

    words = line.split()

    numLines += 1

    # Using len to count the words and characters including spaces

    numWords += len(words)

    numChars += len(line)



# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Opening the data set

ft = open('/kaggle/input/a2data/sherlock.txt', 'r')



# Defining variables to hold total lines, words and characters

num_lines = 0

num_words = 0

num_chars = 0



# Code for printing the file output

# for line in ft:

#    line = line. rstrip('\n')

#    print(line)



# Reading the file line by line

for line in ft:

    # Tracking lines using rstrip

    lines = line.rstrip('\n')

    # Tracking words using split

    words = line.split()

    

    num_lines += 1

    # Sum up words and characters using len()

    num_words += len(words)

    num_chars += len(line)



# Printing the output

print(f'characters {num_chars}')

print(f'lines {num_lines}')

print(f'words {num_words}')

    

# Closing the file

ft.close()