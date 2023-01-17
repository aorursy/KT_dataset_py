mylist = [1, 2, 3, 4, 5, 6]

for i in mylist:

    print(i)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for i in mylist: 

    total = total + i

print(total)
s = "This is a test string for HCDE 530"

# Add your code below

words = s.split()

for word in words:

    print(word)
# Add your code here

mylist[3] = "four"

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Add your code below

f = open('/kaggle/input/test.txt', 'r')

for line in f:

    line = line.rstrip()

    print(line)



# It's good practice to close your file when you are finished. This is in the next line.

f.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for items in dogList:

    if items.find("terrier") > 0:

        print(items.find("terrier"))

    else:

        print(items.find("Terrier"))
binList = [0,1,1,0,1,1,0]



# Add your code below

for i in binList:

    if i == 1:

        print ('One')
for items in dogList:

    if items.lower().find('bulldog') >= 0:

        print(items)
numChars = 0

numLines = 0

numWords = 0



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

f = open('/kaggle/input/test.txt', 'r')

for line in f:

    numChars = numChars + len(line)

    numLines = numLines + 1

    numWords = numWords + len(line.split())



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

f.close()
# Add your code below

numChars = 0

numLines = 0

numWords = 0



f = open('/kaggle/input/sherlock.txt', 'r')

for line in f:

    numChars = numChars + len(line)

    numLines = numLines + 1

    numWords = numWords + len(line.split())



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



f.close()