mylist = [1, 2, 3, 4, 5, 6]

for number in mylist:

    print(number)

    



# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for number in mylist:

    total = total + number

    

print(total)

s = "This is a test string for HCDE 530"

# Add your code below

for word in s.split():

    print(word)

# Add your code here

mylist[3] = 'four'

print(mylist)

# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file



fname = open('/kaggle/input/a2data/test.txt', 'r')

for line in fname:

    x = line.rstrip()

    print(x)    



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



for string in dogList:

    string = string.lower()

    if "terrier" in string:

        print(string.find("terrier"))

    else:

        print(int(-1))

        

        
binList = [0,1,1,0,1,1,0]



for num in binList:

    if num == 1:

        print("One")
for string in dogList:

    if string.find("Bulldog" or "bulldog") > -1:

        print(string)



    
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    numChars = len(line) + numChars

    numLines = numLines + 1

    numWords = len(line.split()) + numWords

    



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
numChars = 0

numLines = 0

numWords = 0



fname = open('/kaggle/input/a2data2/sherlock.txt', 'r')



for line in fname:

    x = line.rstrip()

    numChars = len(x) + numChars

    numLines = numLines + 1

    numWords = len(x.split()) + numWords

    

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)

    

fname.close()