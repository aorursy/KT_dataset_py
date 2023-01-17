mylist = [1, 2, 3, 4, 5, 6]

mylist = [1, 2, 3, 4, 5, 6]
# Define a variable a to iterate over each element of mylist
for a in mylist:
    # Printing each element on a new line 
    print(a)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0
for i in mylist:
    total= total + i
print(total)

# Define accumulator variable total
total = 0
# Iterate through the items in the sequence
for a in mylist:
    # Update the accumulator variable each time 
    total = total + a 
# Print final accumulator variable result
print(total)
s = "This is a test string for HCDE 530"
# Add your code below
# Create a list of separate words separated by blank space
words = s.split()
# Iterate through the items in the list & print each word one by one
for a in words:
    print(a)


# Use the enumerate method to find and replace the element with the value four 
for n, a in enumerate(mylist):
    if a == 4:
        mylist[n] = 'four'
print (mylist)
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below
for line in fname:
    print(line.rstrip())

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()

dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below
for dog in dogList: 
    #if we gound terrier then whe know the character number where that starts
    result = dog.find('terrier')
    if result == -1:
        result = dog.find('Terrier')
    print(dog, result)
binList = [0,1,1,0,1,1,0]

for a in binList:
    if a == 1:
        print('One')

# Define a variable word to input the string to detect and print 
word = "Bulldog"

# Using conditionals with find and lower method 
for dog in dogList: 
    if dog.lower().find(word.lower()) != -1:
        print(dog)
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
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
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below