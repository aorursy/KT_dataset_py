mylist = [1, 2, 3, 4, 5, 6]
for x in mylist: 
    print(x)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0
Sum = sum(mylist) 
print(Sum) 

s = "This is a test string for HCDE 530"

# Add your code below

mylist2 = s.split(" ")
for l in mylist2:
    words = l.split
    print(l)


mylist = [1, 2, 3, 4, 5, 6]
for n, i in enumerate(mylist):
     if i == 4:
        mylist[n] = "four"

mylist


# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/testtxt/test.txt','r')


# Add your code below
for line in fname:
    print(line.strip())

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

for x in dogList:
    print(x.lower().find("terrier"))
    

binList = [0,1,1,0,1,1,0]

# Add your code below

for i in (binList): 
    if i == 1:
        print (i, "one")
    else:
        print (i,"")
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

for x in dogList:
    if (x.lower().find("bulldog")!=-1):
        print(x)
  
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/testtxt/test.txt','r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
    numLines += 1
    print(numLines)
    numChars = numChars + len(line)
    wordslist = line.split()
    numWords = numWords + len(wordslist)

# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
fname = open('/kaggle/input/testtxt/sherlock.txt','r')

fname.close()