mylist = [1, 2, 3, 4, 5, 6]

for item in mylist:

    print(item)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for item in mylist:

    total = total + item

print(total)

s = "This is a test string for HCDE 530"

# Add your code below

words = s.split()

for word in words:

    print(word)

# Add your code here

for item in range(len(mylist)):

    if(mylist[item] == 4):

       mylist[item] = 'four'

print(mylist)



# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

for line in fname:

    cleanline = line.rstrip("\n")

    print(cleanline)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for dog in dogList:

    lowerDog = dog.lower() #changing each dog to lowercase so I can just look for terrier once

    num = lowerDog.find('terrier')

    print(num)
binList = [0,1,1,0,1,1,0]



# Add your code below

for item in binList:

    if item:

        print("One")

for dog in dogList:

    doglower = dog.lower() #switching each string to lowercase so bulldog string only has to be searched for once

    num = doglower.find("bulldog")

    if num > -1: #if bulldog is not found it will be -1, so check for all number great then

        print(dog)

    
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    numChars = numChars + len(line) #for each line, add the character number to the total character count

    numLines = numLines + 1 #for each line looped through, add 1 to the numLines counter

    words = line.split()

    numWords = numWords + len(words) #for each line, add the number of words to the words total



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below

numChars = 0

numLines = 0

numWords = 0

# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



for line in fname:

    numChars = numChars + len(line) #for each line, add the character number to the total character count

    numLines = numLines + 1 #for each line looped through, add 1 to the numLines counter

    words = line.split()

    numWords = numWords + len(words) #for each line, add the number of words to the words total

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()