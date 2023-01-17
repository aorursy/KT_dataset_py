mylist = [1, 2, 3, 4, 5, 6]



#Iterates through mylist to print out each element on a new line

for xyz in mylist:

    print(xyz)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



#Iterates through mylist to sum each value of the list 

for x in mylist:

    total = total + x

print(total)

s = "This is a test string for HCDE 530"

# Add your code below



#Splits the list into separate words 

st = s.split()



#Iterates through the list of words and prints each word out on new line

for x in st:

    print(x)

# Add your code here

index = 0



#Iterates through the list with a conditional IF to find a integer 4 and replace with the string 'four'.

for y in mylist:

    if y == 4:

        mylist[index] = "four"

    #Increments through the list

    index = index + 1

print(mylist)

    
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

#Iterates through the file to read through and print each line.

for line in fname:

    #Removes the newline character at the end of each line

    line = line.rstrip()

    print(line)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

#Iterates through the list with conditional IF and ELIF to identify cases of 'terrier' and 'Terrier'

for x in dogList:

    #When a 'Terrier' is found, print the character number on a new line where the string starts

    if x.find("Terrier") != -1:

        print(x.find("Terrier"))

    #When a 'terrier' is found, print the character number on a new line where the string starts

    elif x.find("terrier") != -1:

        print(x.find("terrier"))

    else:

        print(-1)
binList = [0,1,1,0,1,1,0]

# Add your code below

#Iterate through the list to find instances of integer 1 and print the string 'One'on a new line

for y in binList:

    if y == 1:

        print("One")

#Iterate through the list to print out the instances of 'Bulldog' and 'bulldog' on a new line

for z in dogList:

    if z.find("Bulldog") != -1:

        print(z)

    elif z.find("bulldog") != -1:

        print(z)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#Iterate through the file

for a in fname:

    #Each iteration denotes a separate line

    numLines = numLines + 1

    

    #For testing character count with newline characters included

    #numChars = len(a) + numChars  

    

    #The sum of characters in each line denotes the number of characters with newline characters excluded

    numChars = len(a.rstrip()) + numChars

    

    #The sum of the number of words for each line.  Lines are split into words.

    numWords = len(a.split()) + numWords

    



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



fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



#Iterate through the file

for s in fname:

    #Each iteration denotes a separate line

    numLines = numLines + 1

    

    #For testing character count with newline characters included

    #numChars = len(s) + numChars  

    

    #The sum of characters in each line denotes the number of characters with newlines excluded

    numChars = len(s.rstrip()) + numChars

    

    #The sum of the number of words for each line.  Lines are split into words.

    numWords = len(s.split()) + numWords



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



fname.close()

             