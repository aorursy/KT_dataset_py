mylist = [1, 2, 3, 4, 5, 6]
# Iterates mylist and prints the elements.
for i in mylist:
    print(i)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0
# Add each element in mylist to the total.
for i in mylist:
    total+=i
print(total)

s = "This is a test string for HCDE 530"
# Add your code below
# My list of separate strings
slst = s.split()
# Prints the element of the separeted list of strings
for i in slst:
    print(i)
    
# Add your code here

# Delete the 4(index 3)
del(mylist[3])
# Insert the test "four" into the deleted index
mylist.insert(3,'four')

print(mylist)

# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# print(os.listdir('../input'))
# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below

# Print each line in the file and remove the end of line character
for line in fname:
    print(line.rstrip('\n'))

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

# Iterates over dogList and prints the starting position of the word "terrier" (case insentive)
for i in dogList:
    print(i.lower().find('terrier'))
    # Another way to do it would be.
    # pos = i.find('terrier')
    # if pos != -1:
    #    print(pos)
    # else:
    # print(i.find('Terrier'))
binList = [0,1,1,0,1,1,0]

# Add your code below

# Iterates through binList prints "One" when the element equals 1 and "Infinity" otherwise.
for i in binList:
    if(i == 1): # Given the list values. "if(i)" would have also worked but given that the instructions specify "Otherwise", I decided to include the comparison.
        print("One")
    else:
        print("Infinity")
# Iterates over dogList and prints the value of the names that contain the word "bulldog" (case insentive)
for i in dogList:
    if(i.lower().find('bulldog') != -1):
        print(i)
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
    numLines += 1
    numChars += len(line) # I could change this to "len(line.rstrip("\n"))" to exclude newlines, but it wouldn't match the count above(22,3,5).
    numWords += len(line.split())

# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below
numChars = 0
numLines = 0
numWords = 0

# Create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2data/sherlock.txt', 'r')

# Read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
    numLines += 1
    numChars += len(line.rstrip("\n"))
    numWords += len(line.split())

# Print the result
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()