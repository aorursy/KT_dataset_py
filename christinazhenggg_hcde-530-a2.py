mylist = [1, 2, 3, 4, 5, 6]



# x runs through every string on the list one by one 

for x in mylist:

    print (x)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



#total the item on the list one by one

for i in mylist:

    total = total + i



print (total)

    

s = "This is a test string for HCDE 530"

# Add your code below



# split s into a list with individual items

for x in s:

    words = s.split()



# loop through the list 'words' and print the strings one by one 

for y in words:

    print (y)  
# Add your code here



# use indexing to change 3rd item on the list to 'four'

mylist [3] = 'four'



# print the list 

print (mylist)

# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

for x in fname:

    print (x)

    



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



#set a for loop to go through each string

for name in dogList:

    x = name.find("terrier")

    y = name.find("Terrier")  

#set a condition that select out strings that contain 'terrier' or 'Terrier' first, print others as '-1'

    if x>-1:

        print (x)

    elif y>-1:

        print (y)

    else:

        print ('-1')





binList = [0,1,1,0,1,1,0]



# Add your code below



for x in binList:

#if x has the value of 1, then print 'One'

    if x == 1:

        print ("One")
#set a for loop to go through each string

for name2 in dogList:



#find ones with bulldog in its name

    a = name2.find("Bulldog")

    if a > -1:

        print (name2)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for lines in fname:

    numLines += 1

    numChars += len(lines)

    words = lines.split()

    numWords += len(words)



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

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for lines in fname:

    numLines += 1

    numChars += len(lines)

    words = lines.split()

    numWords += len(words)



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()