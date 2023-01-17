mylist = [1, 2, 3, 4, 5, 6]

for x in mylist:

    print(x)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0



for stepTwo in mylist:

    total= total+stepTwo

print(total)



s = "This is a test string for HCDE 530"

# Add your code below

s= s.split()

for testString in s:

    print(testString)

# Add your code here

y=0

for x in mylist:

    if x == 4:

        y = x-1

        mylist[y]= 'four'

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

    print(line)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

# Add your code below

for x in dogList:

    x=x.lower()

    y=x.find('terrier')

    print("result for", x, "=",y)
binList = [0,1,1,0,1,1,0]



# Add your code below

for x in binList:

    if x== 1:

        print('One')
for x in dogList:

    y=x.find('Bulldog')

    x = x.lower()

    if y != -1:

        print(x)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



fname = open('/kaggle/input/a2data/test.txt', 'r')

lines = fname.readlines()

numLines=len(lines)



for line in lines:

    numChars += len(line)

    numWords += len(line.split())



#Joel's (long suffering) attempt

#eachLine = []

#charCount= []

#worCount= []



#for line in fname:

#    eachLine.append(line)

#    for x in line:

#        charCount.append(x)

    

#numLines= len(eachLine)

#numChars=len(charCount)





# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below

sherlock= open('/kaggle/input/sherlock/sherlock.txt', 'r')



lines = sherlock.readlines()

numLines=len(lines)



for line in lines:

    numChars += len(line)

    numWords += len(line.split())



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)