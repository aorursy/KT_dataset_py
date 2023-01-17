#define mylist

mylist = [1, 2, 3, 4, 5, 6]



#set up loop to iterate through 

for x in mylist: #In this case the variable name x is bound to each item in mylist

    print (x) #go through mylist and print x each time it iterates 
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



#define total

total = 0



#set up loop to iterate through

for x in mylist: #In this case the variable name x is bound to each item in mylist

    total = total + x #instruct total to add x in each iteration



print (total) #print final total
s = "This is a test string for HCDE 530"

# Add your code below



for word in s.split(): #set up a for loop to iterate through. In this case the variable name word is bound to each item in string s

#s.split indicates the code should split the string each time there is a space. 

#Therefore, spaces separate each item (named word) in the string

    print (word) #print each item in the string
# Add your code here

mylist [3] = 'four' #redefine 4th item in mylist as four



#print out new version of mylist

print (mylist) 
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

for line in fname:#set up a for loop to iterate through the variable name line is bound to each item in file filename

    print(line.rstrip())#get rid of extra new lines



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
#define variables

dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

indexPosition = 0



#set up a for loop to iterate through the variable name dog is bound to each item in dogList

for dog in dogList:

    dog = dog.lower() #convert doglist to be all lower case

    indexPostion = dog.find("terrier") #define indexposition. We are only interested in finding the index when dog = terrier

    print(indexPostion) #print out index 
binList = [0,1,1,0,1,1,0]



# Add your code below

for n in binList: #set up a for loop to iterate through the variable name n is bound to each item in binList

    if n ==1: #if n variable is one print "one"

        print("One")

    else:#otherwise print nothing.

        print()
#define variables

dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

indexPosition = 0



#set up a for loop to iterate through the variable name dog is bound to each item in dogList

for dog in dogList:

    dog = dog.lower() #convert doglist to be all lower case

    #assign insdexPosition variable to exist when bulldog is found to be part of varibale dog 

    indexPosition = dog.find("bulldog")

   

    if indexPosition >= 0: #look for this specific casr in the loop

    #indexPosition -1 indicates item was not found

    #so we want only indexPositions greater than or equal to zero  

        print(dog) #print full dog variable if found

    else:#otherwise print nothing.

        print()
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#set up loop to iterate through

for x in fname: #x is bound to each item in fname

    numChars = len (x) + numChars 

    #character count = adding the length of x, which in this case is the number of characters per line,to each iteration 

    #esentially we are adding each character count to itself as it goes through each line. 

    

    numLines = numLines + 1

    #line count = add one each time we iterate. 

    #The number of times we iterate is dictated by line number, so this is an easy way to count our lines 

    

    word = x.split() #split lines into words

    numWords =  len (word) + numWords

    #word count = counting how many words are in each line, and adding them to the pervious iteration 



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
numChars = 0

numLines = 0

numWords = 0



# create a file handle called file to open and hold the contents of the data file

file = open('/kaggle/input/sherlocka2/sherlock.txt','r')



#set up loop to iterate through

for x in file: #x is bound to each item in fname

    numChars = len (x) + numChars 

    #character count = adding the length of x, which in this case is the number of characters per line,to each iteration 

    #esentially we are adding each character count to itself as it goes through each line. 

    

    numLines = numLines + 1

    #line count = add one each time we iterate. 

    #The number of times we iterate is dictated by line number, so this is an easy way to count our lines 

    

    word = x.split() #split lines into words

    numWords =  len (word) + numWords

    #word count = counting how many words are in each line, and adding them to the pervious iteration 



# output code taken from pervious example

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

file.close()