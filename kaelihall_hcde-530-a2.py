mylist = [1, 2, 3, 4, 5, 6]

for x in mylist:#this will create a finite loop of mylist and print its components

    print(x)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:#This will create a finite loop of mylist

    total=total+x#This will create a compounding sum of the numbers in mylist

print(total)
s = "This is a test string for HCDE 530"

# Add your code below

words=s.split()#This splits the string into individual words

for x in words:#This creates 'x' loop of the words into their own lines

    print(x)
# Add your code here

print(mylist)#This displays the original list

r=0#This creates the baseline to later identify which space "4" occupies

for x in mylist:#This creates a loop of "mylist"

    if x==4:#This identifies the spot of the original "4"

        banana=r#This creates a variable to mark the location of the "4"

    r=r+1#This lines up the variable with the space that "4" occupies to account for the 0th spot

mylist[banana]="four"#This replaces the item in the spot marked by variable "banana" with the string "four"

print(mylist)#This prints the altered list



# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/testtxt/test.txt', 'r')#This asks the program to open and read the test.txt file



# Add your code below



for r in fname:#This loops the text within the test.txt file as defined in "fname"

    print(r)#This prints the loop



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for xy in dogList:

    result = xy.find('terrier')#This searches the list to identify if the string includes the word 'terrier'

    result2=xy.find('Terrier')#This does the same as above for "Terrier"

    if result!=-1:#This looks through the result of the find command, and if the find command doesn't return a "-1" (-1 would mean that the string does NOT include the word 'terrier') to follow the next command

        print(result)#This will print the location of the occurence from the above find command

    elif result2!=-1:#This does the same as the if command but for "Terrier"

        print(result2)

    else:#This tells the program what to do for the strings which do not include either "terrier" or "Terrier"

        print("-1")

binList = [0,1,1,0,1,1,0]



# Add your code below



for b in binList:#This creates a loop of binList

    if b==1:#This searches for the result of the loop which is equal to 1

        print("One")#This tells the program to print the string "One" following the above criteria
for dog in dogList:#This creates a loop for the dogList

    bull=dog.find("Bulldog")#This searches for the strings which include "Bulldog"

    if bull!= -1:#This tells the program that if the result of the find command does not return a -1 and therefore does include "Bulldog" it should follow the following print command

        print(dog)#This tells the program to print the list item
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/testtxt/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for p in fname:

    print(p)

    numChars=numChars+len(p)#This defines numChar as the total length of the file characters

    nolines=p.rstrip("\n \n")#This removes the lines which are blank from loop

    n=nolines.split("\n")#This splits the file into strings defined by the separation of lines

    numLines=numLines+len(list(n))#This defines numLines as the length of the list of strings defined above

    l=p.split(" ")#This splits the strings into words as separated by spaces

    numWords=numWords+len(l)#This defines numWords as the number of strings split as defined above

    



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below



fname2=open('/kaggle/input/sherlocktxt/sherlock.txt', 'r')



for sherlock in fname2:

    numChars=numChars+len(sherlock)#This defines numChar as the total length of the file characters

    nolines=sherlock.rstrip("\n \n")#This removes the lines which are blank from loop

    n=nolines.split("\n")#This splits the file into strings defined by the separation of lines

    numLines=numLines+len(list(n))#This defines numLines as the length of the list of strings defined above

    l=p.split(" ")#This splits the strings into words as separated by spaces

    numWords=numWords+len(l)#This defines numWords as the number of strings split as defined above

    



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



fname2.close()