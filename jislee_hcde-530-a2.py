mylist = [1, 2, 3, 4, 5, 6]

for line in mylist:

    print(line) #print each item on a new line

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:

    total+=x #add up the numbers in mylist

print(total)
s = "This is a test string for HCDE 530"

# Add your code below

x=s.split() #splits the line into words

for words in x:

    print (words) #print each word on a new line
# Add your code here



newlist=[]#creating an empty list



for i in mylist:

    if i==4: #check to see if there is "4" in myist

       i='four'#if so, replace it with 'four'

    newlist.append(i) #add the items in the newlist

mylist=newlist #make newlist the same as mylist

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

    print (line.rstrip()) #print the text without the new line



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

doglist=[item.lower() for item in dogList] # change all the items in the list to lowercase

# Add your code below

for i in doglist:

    x=i.find("terrier") #find the position 'terrier'starts with in each item

    print (x)

   

    

binList = [0,1,1,0,1,1,0]



# Add your code below

for i in binList:

    if i==1:

        print (i,"One") #prints 'one' every time there's a '1'in the list

    else:

         print (i)



        
for i in dogList:

    x=i.find("Bulldog") # x= the position Bulldog starts in each item

    if x>=0:

        print (i)

    
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    numLines+=1 #adds the number of lines

    words=line.split() #split the line into words

    numWords+=len(words) #counts number of words in a line

    numChars+=len(line) # counts number characters in a line







# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below

numChars=0

numWords=0

numLines=0



fname2=open("/kaggle/input/a2data2/sherlock.txt",'r')

for line in fname2:

    numLines+=1 #adds the number of lines

    words=line.split() #split the line into words

    numWords+=len(words) #counts number of words in a line

    numChars+=len(line) # counts number characters in a line



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



fname2.close()