mylist = [1, 2, 3, 4, 5, 6]

for xyz in mylist:

    print(xyz)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



#creating a variable to be used to sum values

total = 0

#iterating through mylist, adding to the variable "total" during every loop

for x in mylist:

    total=total+x

#print result

print(total)

s = "This is a test string for HCDE 530"

# Add your code below



# creating variable "w" for words that are split from "s" by spaces

w=s.split()



#iterate over "w" and printing the results for each loop as "a"

for a in w:    

    print(a)



# Add your code here



#since mylist is a list of sequential numbers increasing by 1, 

#we can reflect the value of each element in the list by looping through the list and adding 1 each time

#creating a variable "index" to sum through mylist

index=0

#iterating over mylist and adding 1 each loop

for v in mylist:

    index=index+1

#identifying the 4th loop of mylist, and assigning "position" to equal to the "index" value which is 4     

    if v==4:

        position=index  

#changing 4 to "four"

mylist[position]="four"

print(mylist)

    

# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

#Remove newline

for line in fname:

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

#iterate over dogList to find substrings that matches 'terrier'

for line in dogList:

    result = line.find('terrier')

#print code with result and dog names

    print (result,line) 





    
binList = [0,1,1,0,1,1,0]



# Add your code below



for line in binList:

    if line==1:

        print("One")

   

    
for line in dogList:

    result = line.find('Bulldog')

    if result>-1:

        print(line)

numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')





# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#iterate over file and split it by spaces forming a list of words

for line in fname:

    words=line.split()

#number of lines equals to the number of "line"loops

    numLines = numLines+1

#number of words equals to the sums of length of the words lists each line

    numWords = numWords+len(words)

#number of charaters equals to the sums of lengths of each line

    numChars = numChars+len(line)



   

    

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

for line in fname:

    words=line.split()

    numLines = numLines+1

    numWords = numWords+len(words)

   #counting characters excluding newline characters

    numChars = numChars+len(line.rstrip())



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



fname.close()