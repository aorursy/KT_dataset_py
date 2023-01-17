mylist = [1, 2, 3, 4, 5, 6]

for element in mylist:

    print(element)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for element in mylist:

    total += element



print(total)
s = "This is a test string for HCDE 530"

# Add your code below



word = ''

for char in s:

    if char == ' ':

        print(word)

        word = ''

    else:

        word = word + char

        

if word != '':

    print(word)
# Add your code here



index = 0

for element in mylist:

    if element == 4:

        mylist[index] = "four"

    index += 1

    

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

line = fname.readline()

while line:

    print(line.rstrip())

    line = fname.readline()



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



for dog in dogList:

    if "terrier" in dog or "Terrier" in dog:

        print(dog.find("terrier"))

    else:

        print(-1)

binList = [0,1,1,0,1,1,0]



# Add your code below



for num in binList:

    if num == 1:

        print("One")

for dog in dogList:

    if "bulldog" in dog.lower():

        print(dog)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



line = fname.readline()

while line:

    numLines += 1

    for word in line.rstrip().split(' '):

        numWords += 1

        if word[-1] == ',' or word[-1] == '.':

            numChars += (len(word) - 1)

        else:

            numChars += len(word)

    line = fname.readline()



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



line = fname.readline()

while line:

    numLines += 1

    for word in line.rstrip().split(' '):

        numWords += 1

        # filter out symbols

        while len(word) > 0 and not word[-1].isalpha() and not word[-1].isnumeric():

            word = word[:-1]

        numChars += len(word)

    line = fname.readline()



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()