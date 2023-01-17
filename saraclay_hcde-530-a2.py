mylist = [1, 2, 3, 4, 5, 6]



# Printing out each number one by one in a for-loop

for x in mylist:

    print (x)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0



# This is showing cumulative addition

for x in mylist:

    total=total+x

    print(total)

s = "This is a test string for HCDE 530"

# Add your code below

y=s.split()



# For each loop, the word that comes out of the variable "y" goes on a new line

for x in y:

    print (x)

# Add your code here

mylist = [1, 2, 3, 4, 5, 6]



# This will replace the fourth item in this list with "four"

mylist[3]="four"

print (mylist)





# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/testtext/test.txt', 'r')



# Add your code below

for f in fname:

    # This will print out each line

    print (f.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for dogs in dogList:

    # If the word "Terrier" (uppercase) is in this item, print the character number of where it starts

    if (dogs.find("Terrier") != -1):

        print (dogs.find("Terrier"))

     # If the word "terrier" (lowercase) is in this item, print the character number of where it starts

    elif (dogs.find("terrier") != -1):

        print(dogs.find("terrier"))

    # If neither "Terrier" nor "terrier" are in the item, just print -1

    else:

        print(-1)

                                               

binList = [0,1,1,0,1,1,0]



# Add your code below

for num in binList:

    # If the item in the list is "1" then print "One"

    if num==1:

        print("One")

for dogs in dogList:

    # If the item in dogList has "Bulldog" then print the name of the dog. Otherwise, don't print anything.

    if (dogs.find("Bulldog") != -1):

        print(dogs)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first

fname = open('/kaggle/input/testtext/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for f in fname:

    

    # Cumulative addition of the length of each word

    numChars=len(f)+numChars

    # Cumulative addition of the first letter in each line

    numLines=len(f[0])+numLines

    # Cumulative addition of items in each line's list

    numWords=(len(f.split()))+numWords

    

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

# make sure to upload your test.txt file first

fname = open('/kaggle/input/reading/sherlock.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for f in fname:

    

    # Cumulative addition of the length of each word

    numChars=len(f)+numChars

    # Cumulative addition of the first letter in each line

    numLines=len(f[0])+numLines

    # Cumulative addition of items in each line's list

    numWords=(len(f.split()))+numWords

    

# output code below is provided for you; you should not edit this



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()