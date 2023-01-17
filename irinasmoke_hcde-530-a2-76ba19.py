mylist = [1, 2, 3, 4, 5, 6]



#iterate over list and print

for i in mylist:

    print(i)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



#accumulation pattern

for i in mylist: 

    total += i

    

print(total)
s = "This is a test string for HCDE 530"

# Add your code below



#split string into list

list=s.split() 



#iterate over list and print each item

for i in list:

    print(i)
# Add your code here



#find the value 4 in the list and set that element to be "four" instead

i = mylist.index(4) 

mylist[i]="four" 



print (mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

# Iterate over the file. Strip spaces from the beginning and end of each line and print it.

for line in fname:

    print(line.strip()) 



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()

dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



#Iterate over each item in dogList

for str in dogList:

    a =str.find("terrier") #index of "terrier"

    b =str.find("Terrier") #index of "Terrier"

    

    #If str contains "terrier," print the index of the start of the word

    if (a != -1):

        print(a)

    #If str contains "Terrier," print the index of the start of the word

    elif (b != -1):

        print(b)

    else:

        print("-1")  #else print -1
binList = [0,1,1,0,1,1,0]



# Add your code below

for i in binList:

    #if the current item equals 1, print "One"

    if i==1:

        print("One")

#iterate over list

for i in dogList:

    if "Bulldog" in i or "bulldog" in i: #find "bulldog" or "Bulldog"

        print(i)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#iterate over lines

for line in fname:

    numLines += 1 #increment numLines

    

    #Get the number of characters in the line and add to numChars

    chars = len(line)

    numChars += chars

    

    #Get the number of words in the line and add to numWords

    words = line.split()

    numWords += len(words)



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



# Read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#iterate over lines

for line in fname:

    numLines += 1 #increment numLines

    

    #Get the number of characters in the line and add to numChars

    chars = len(line)

    numChars += chars

    

    #Get the number of words in the line and add to numWords

    words = line.split()

    numWords += len(words)



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()