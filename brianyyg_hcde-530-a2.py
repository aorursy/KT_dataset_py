mylist = [1, 2, 3, 4, 5, 6]

# loop through print all items using a for loop
for item in mylist:
    print(item)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0

# create a for loop
for item in mylist:
    
    # add up first 5 items
    if item < 6:
        total = total + item
        
    # when it comes to the last item, add it to the total, then print the result
    else:
        total = total + item
        print (total)
    

s = "This is a test string for HCDE 530"
# Add your code below

# split strings in s to a list of seperated strings
sp = s.split()

# use a for loop to put each string on a single line
for word in sp:
    print (word)
# Add your code here

# replace 4 with four with indexing
mylist[3] = "four"

#print to check the result
print (mylist)
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/assignment-2-test-file/test.txt', 'r')

# Add your code below
# print all lines and remove extra blank lines with strip()
for line in fname:
    print (line.strip())

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

# loop through the list
for le in dogList:
    
    # identify if an item has "terrier" in it.
    if "terrier" in le:
        
        # use find function to point out the number of the word, and print it out
        print (le, le.find("terrier"))
    
    # identify if an item has "terrier" in it.
    elif "Terrier":
        
         # use find function to point out the number of the word, and print it out
        print (le, le.find("Terrier"))
        
    # if can't find the demanded word, print out "-1"
    else:
        print (le, -1)        
binList = [0,1,1,0,1,1,0]

# Add your code below

# Loop through items
for z in binList:
    
    #Identify "1" and print "One"
    if z == 1:
        print ("One")
        
    # otherwise, leave a blank line without anything in it
    else:
        print("\n")
# loop through dogList
for dg in dogList:
    
    # print out items that contain the string Bulldog
    if "Bulldog" in dg:
        print (dg)
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/assignment-2-test-file/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.

# set up a for loop
for line in fname:
    
    # split each line to seperate words and nest in a new variable
    wordList = line.split()
    
    # as each line looped through, add 1 to the number of lines
    numLines += 1
    
    # as each line looped through, add number of words
    numWords += len (wordList)
    
    # line is a string, the len of this string is the number of characters
    numChars += len (line)

# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below

# create placeholders
numChars = 0
numWords = 0
numLines = 0

# access data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle
file = open('/kaggle/input/a2data/sherlock.txt','r')

# use for loop to calculate the total of each elements
for line in file:
    wordList = line.split()
    numLines += 1
    numWords += len(wordList)
    numChars += len(line)
    
# print out the results
print ("lines "+ str(numLines))
print ("words "+ str(numWords))
print ("Characters "+ str(numChars))