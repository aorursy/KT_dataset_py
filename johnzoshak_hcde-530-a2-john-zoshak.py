mylist = [1, 2, 3, 4, 5, 6]



for x in mylist: #takes each item in the list and iterates through them one by one. Python is magic. 

    print(x) ## prints inside the loop so as x takes on the value of each iteration it'll print x. 



# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist: #iterates through the list same as above

    total = total + x #takes the total variable and adds each iteration to it as the for loop is running. 

print(total) #prints the total

s = "This is a test string for HCDE 530"

# Add your code below



splitted = s.split() #stores my split string into a variable. Splits on a space. Stores each split as an item in a list. 



for x in splitted: #iterates through my list and prints out each item.

    print(x)



# Add your code here

mylist = ["this", 4, "seven", "eight"]

mylist[3] = "four" #changes the 4th item to the value "four" from 4. Starts at 0 hence 3

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

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



for dog in dogList:

    if (dog.find("terrier") != -1): # checks to see if terrier is in the string

        print(dog.find("terrier")) #prints the where the character is in the string

    elif (dog.find("Terrier") != -1): # checks to see if terrier is in the string.

        print(dog.find("Terrier"))



binList = [0,1,1,0,1,1,0]



# Add your code below



for number in binList:

    if(number == 1): #checks to see if the current iteration is equal to one.

        print("one")# prints one

for dog in dogList: 

    if(dog.find("bulldog") != -1): # checks to see if bulldog is in dog.

        print(dog)

    elif(dog.find("Bulldog") != -1): #checks to see if Bulldog is in dog.

        print(dog)

numChars, numLines, numWords = (0,0,0)



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    numChars = len(line.rstrip()) + numChars #takes the length of the line excluidng new line characters

    numWords = len(line.split()) + numWords #Splits the line into a list of words and then takes the number of items in that list which is the number of words

    numLines = numLines + 1 #basic counter that counts the number of lines that is hacky but I think works? 



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below

fname = open('/kaggle/input/sherlock/sherlock.txt', 'r')



numChars = 0 #sets up a variable for later use

numLines = 0 # ditto

numWords = 0 # ditto



for line in fname:

    numChars = len(line.rstrip()) + numChars #takes the length of the line excluidng new line characters; same code as above

    numWords = len(line.split()) + numWords #Splits the line into a list of words and then takes the number of items in that list which is the number of words; same code as above

    numLines = numLines + 1 #basic counter that counts the number of lines that is hacky but I think works?; same code as above



print('%d characters'%numChars) #prints the number of characters; same code as above

print('%d lines'%numLines) #prints the number of characters; same code as above

print('%d words'%numWords) #prints the number of characters; same code as above

    

fname.close()