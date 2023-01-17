mylist = [1, 2, 3, 4, 5, 6]

for exc1 in mylist:
    print(exc1)

# added a colon at the end of for loop and used print to print each element on new lines
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0   #starts from 0

nums = mylist  #mylist brings [1,2,3,4,5,6]
for x in nums:
    total = total + x   #numbers add up on each loop

print(total)
s = "This is a test string for HCDE 530"
# Add your code below

#have a new string that has words splited
newstring = s.split() 
#use for loop to iterate
for line in newstring: 
    #print each word in lines
    print(line)
#using indexing
del(mylist[3]) #delete 4 on the fourth place
mylist.insert(3, 'four') #add four on the fourth place

print(mylist)
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/test.txt', 'r')
# I did updated my file under a2data, but it works when I use the url without a2data

# Add your code below
for line in fname:
    final = line.rstrip() #use restrip() to remove the extra blank line
    print(final)

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

#iterate each elements
for word in dogList:
    
    result = word.find('Terrier') #find names with Terrier, this will let names with Terrier result in character numbers, and other will result in -1
    
    if result == -1: #among the names resulting in -1
        result = word.find('terrier') #find names with terriers to result in character numbers 
    
    print(word, result) #print dog names (word) and resulting numbers
binList = [0,1,1,0,1,1,0]

# Add your code below

for line in binList: #iterate over binList
    if line == 1: #if the item equals 1
        print(line, 'one') #print "one"
    else: #otherwise, do not print anything
        print(line) #print the strings only

dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

#define Bulldog
word = "Bulldog"

#iterate over dogList
for bulldog in dogList:
    #use if to find Bulldog, upper() is to find words regardless of lower or upper cases.
    #find function results in value -1 for words without Bulldog, so used "not equal" to find words with Bulldog
    if bulldog.upper().find(word.upper()) != -1: 
        print(bulldog)
    
    
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
        words = line.split() #split each word
        numWords += len(words) #count the number of words
        
        lines = line.rstrip() #delete the extra blank lines by using rstrip()
        numChars += len(lines) #count the number of characters
        
        numLines += 1 #count number of lines
        
# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below

# open the file sherlock.txt
fname2 = open('/kaggle/input/sherlock.txt', 'r')

#starts from value 0
numChars2 = 0
numLines2 = 0
numWords2 = 0

#iterate with for loop
for sherlock in fname2:
    words = sherlock.split() #every words are on new lines
    numWords2 += len(words) #count the number of words
    
    lines = lines.rstrip() #get rid of extra blank lines
    numChars2 += len(lines) #count the number of characters on each line and sum up
    
    numLines2 += 1 #count the number of lines

#print the result
print(f'characters {numChars2}')
print(f'words {numWords2}')
print(f'lines {numLines2}')

    
#close the file
fname2.close()