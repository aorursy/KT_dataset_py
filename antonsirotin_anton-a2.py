mylist = [1, 2, 3, 4, 5, 6]

for num in mylist: #assigns each ityem of tbeh list to var num
    print(num) #prints the numberr it's on
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.
total = 0 # sets value of sum
for num in mylist: #iterates through list
    total = total + num #adds each iteration of num to total
    print(total)

s = "This is a test string for HCDE 530"
for word in s.split(): #iterates through each word of the string
    print(word) #prints the word

for x in mylist: #iterate through mylist
    if x == 4: 
        x = 'four' #if x==4 is True, replace it with string
        print(x) #print it
    else:
        print(x) #otherwise just print it
         
    

# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/a2-data/test.txt', 'r')

# Add your code below
for line in fname: #iterates through the entire test.txt file through var 'line'
    print(line.rstrip()) #prints out contents of 'line' sans the /n escapes

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane", "Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below
for dog in dogList:  #iterates through dogList using var dog
    occ = dog.find('terrier') #whenever terrier is found in dog, assign character number to var occ
    if occ == -1: 
        occ = dog.find('Terrier') #whenever dog doesn't contain 'terrier', assign character number of 'Terrier' to occ
    print(dog, occ) #print 
binList = [0,1,1,0,1,1,0]
    
# Add your code below
for item in binList: #iterates through binList with var item
    if item == 1:
        item = 'One'  #if item is 1, replace it with str One
        print(item)  # print results

for item in dogList:  #scans through dogList with var item
    if 'Bulldog' in item:
        print(item) #if item contains 'Bulldog', print item
    elif 'bulldog' in item:
        print(item) #if item doesn't contain 'Bulldog' but contains 'bulldog', print item
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2-data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname: #scan through fname with var line
    numLines += 1 #add 1 to numLines every time you iterate through a line
    words = line.split() #creates list 'words' of all the words in each line
    numWords += len(words) #counts the number of words in a line and adds them to numWords
    numChars += len(line) #add to numChars the number of characters in a line
    

# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below
sherChars = 0
sherLines = 0
sherWords = 0

fname = open('/kaggle/input/a2-data/sherlock.txt', 'r')

for line in fname: #scan through fname with var line
    sherLines += 1 #add 1 to numLines every time you iterate through a line
    words = line.split() #creates list 'words' of all the words in each line
    sherWords += len(words) #counts the number of words in a line and adds them to numWords
    sherChars += len(line) #adds to numChars the number of characters in a line
    
print(f'characters {sherChars}')
print(f'lines {sherLines}')
print(f'words {sherWords}')

fname.close()