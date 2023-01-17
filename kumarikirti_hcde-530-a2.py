mylist = [1, 2, 3, 4, 5, 6]

for x in mylist:

    print(x)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:

    total = total+x

print(total)

s = "This is a test string for HCDE 530"

# Add your code below

for x in s:

    word = s.split()

for y in word:

   print(y)
# Add your code here

mylist[3]="four"

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

   print(line)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for x in dogList:

    if 'terrier' in x:

            i=x.find("terrier")

            print(i)

    if 'Terrier' in x:

            i=x.find("Terrier")

            print(i)
binList = [0,1,1,0,1,1,0]



# Add your code below

for x in binList:

    if x==1:

        print("One")
for x in dogList:

    if 'bulldog' in x:

            print(x)

    if 'Bulldog' in x:

            print(x)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/test.txt', 'r')

for line in fname:

    numLines=numLines+1

    words = line.split()

    for n in words:

        numWords = numWords+1

        char=n.rstrip()

        for nchar in char:

            numChars = numChars+1

# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.







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

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/sherlock.txt', 'r')

for line in fname:

    numLines=numLines+1

    words = line.split()

    for n in words:

        numWords = numWords+1

        char=n.rstrip()

        for nchar in char:

            numChars = numChars+1

# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.







# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()