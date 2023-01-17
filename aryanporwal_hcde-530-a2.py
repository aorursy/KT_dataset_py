mylist = [1, 2, 3, 4, 5, 6]
# Printing each element on a new line
for each in mylist:
    print(each)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0
# defined variable to store the list items
for num in mylist:
    total = total + num
    
print('The sum of the list is:', total)

s = "This is a test string for HCDE 530"
# Add your code below

# defined variable to store the words in the line
sp = s.split()
for words in sp:
    # Printing separated words in new line 
    print(words)
    
# Add your code here
for num in mylist:
    # Using if to detect when value 4 is encountered
    if (num == 4):
        # Replacing '4' with a new value 'four'
        mylist[3] = 'four' 
#Printing the new list
print(mylist)
    
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/test.txt', 'r')

# Add your code below
for line in fname:
    # Removing the extra new line using rstrip
    line = line.rstrip('\n')
    #Printing print each line contained in the file 
    print(line)

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below
for x in dogList:
    # finding and printing the index when 'terrier' or 'Terrier' is encountered
    if(x.find('terrier') != -1):
        print('Position where the word terrier starts in',x,'is at', x.find('terrier'))
    elif(x.find('Terrier') != -1):
        print('Position where the word Terrier starts in',x,'is at', x.find('Terrier'))
    else:
        # Printing '-1' in all other cases
        print('-1')
    
binList = [0,1,1,0,1,1,0]

# Add your code below
for item in binList:
    # Iterating the list and printing 'One' if the current item equals 1
    if(item == 1):
        print('One')
    else:
        continue

for x in dogList:
    # Printing the items that containg the string 'Bulldog' or 'bulldog'
    if(x.find('Bulldog') != -1):
        print(x)
    elif(x.find('bulldog') != -1):
        print(x)
    else:
        continue
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.

for text in fname:
    # Using rstrip to separate the lines
    lines = text.rstrip('\n')
    # Using split to separate the words in a line
    words = text.split()
    numLines += 1
    # Using len to count the words and characters
    numWords += len(words)
    numChars += len(text)

# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below

# Opening the data set
ft = open('/kaggle/input/sherlock.txt', 'r')

# Defining variables to hold total lines, words and characters
num_lines = 0
num_words = 0
num_chars = 0

# Code for printing the file output
# for line in ft:
#    line = line. rstrip('\n')
#    print(line)

# Reading the file line by line
for text in ft:
    # Using rstrip to separate the lines
    lines = text.rstrip('\n')
    # Using split to separate the words in a line
    words = text.split()
    
    num_lines += 1
    # Using len to count the words and characters
    num_words += len(words)
    num_chars += len(text)

# Printing the output
print(f'characters {num_chars}')
print(f'lines {num_lines}')
print(f'words {num_words}')
    
# Closing the file
ft.close()