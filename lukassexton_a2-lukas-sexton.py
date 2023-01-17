mylist = [1, 2, 3, 4, 5, 6]

for mylist in mylist:
    print(mylist)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.
# Accumulation Pattern (Ended up having to look at slides)

#total_amt = sum(mylist)
#for a in mylist:
#    total = total + a 
#print(total)
#for a in mylist:
   # if a > sum_so_far:
      #  sum_so_far = a

# sum_so_far = nums [0]

#Used Python tutor to understand the execution

mylist = [1, 2, 3, 4, 5, 6]
total = 0
nums = mylist
for a in nums:
    total = total + a
    
s = "The sum of mylist is "
print (s + str(total))
# str() converts the intager total into a string
#Inclass Coding 4/23
total = 0
for i in mylist:
    total = total +i
print(total)
    
s = "This is a test string for HCDE 530"
# Add your code below
words = s.split()
    # s.split() will split the string into words
print(words)
for item in words:
    #Python is automatically indexing item++
    print (item)
print(words[3])
#Old work with Errors where i am  trying to manual index
# For Personal Reference, Not to be graded
s = "This is a test string for HCDE 530"
# Add your code below
for a in s:
    words = s.split()
    # s.split() will split the string into words
    print(words[0:8])
    
#Inclass Coding 4/23

s = "This is a test string for HCDE 530"
# Add your code below
word = s.split()
for i in words:
    print(i)

# Add your code here
mylist[3] = 'four'# change the item to four
print(mylist)
#Inclass Coding 4/23
# in order to access 4, we need to access the the third position. 
#mylist[3] = 'four' #change the item to four
#print (mylist)
mylist = [1, 2, 3, 4, 5, 4, 6]
for n, i in enumerate (mylist):
    if i == 4:
        mylist[n] = 'four' # n keeps track of our position
        print(n)
print (mylist)

# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
fname = open('/kaggle/input/a2data/test.txt', 'r')
for line in fname:
    print(line)



#List 
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below
    
for line in dogList:
    #If  line == "terrier" or "Terrier"

#print out the character number where the string "Terrier" or "terrier" starts.
    
    
    if (line.find('terrier') != -1):
        char_num1 = line.index('terrier')
        print (char_num1)
    
    elif (line.find('Terrier') != -1):
        char_num2 = line.index('Terrier')
        print(char_num2)
    
# if the line does not contain "terrier" or "Terrier", your code should print "-1"    
     
    else:
        print (-1)



#OLD CODE for reference 
#for line in dogList:
#    if line == "Terrier":
#        print (line)
     
#    else:
#        print (-1)

# Boolean Check to see if they are within the list
#terrier in dogList
#"Terrier" in dogList

#dogList.index(terrier)

#locate = dogList.split()
#print(locate)

# INCLASS CODING 4/23
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]
for dog in dogList:
    result = dog.find('terrier') # Find must be execusted on an object must be iterated through dog
    if result == -1:
        result = dog.find('Terrier')
    print(dog,result) #Can include a couple of variables
        
binList = [0,1,1,0,1,1,0]

# Add your code below
# Counts through the 6 item list and ends if the variable count is not equal to one
for count in binList:
    if count == 1:
        print("One")
    else:
        False

        
        
        
        #if one in binList:
 #   print(binList)
    
#for line in dogList:
#    if line == "Terrier":
#        print (line)
     
#    else:
#        print (-1)

# INCLASS CODING 4/23

binList = [0,1,1,0,1,1,0]

# Add your code below
for i in binList:
    if i ==1:
        print('One')
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

for line in dogList:
    if line == "Bulldog":
        print (line)
    
    # Supposed to find alternative bulldog so that spelling doesnt matter. 
    elif line == "French Bulldog":
        print (line)
    else:
        False
#OLD CODE FOR REFERENCE 
#import re
#dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]
#x = re.findall("\w", dogList)
#for line in dogList:
#    if line == "\w Bulldog":
#        print (line)
#    elif line == "\w bulldog":
#        print (line)
 #   else:
  #      False
# INCLASS CODING 4/23
word = "Bulldog"

for dog in dogList:
    if dog.lower().find(word.lower()) != -1: # will lowercase all the words in list  # The word.lower feeds into the find boject which feeds into the dog.ower object 
        print (dog)
    # if it not true it doesnt match (double Negativ)
for dog in dogList:
    if word.lower() in dog.lower():
        print (dog)
# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your a2feed.txt file first
# Opens the file named test.txt
fname = open('/kaggle/input/a2data/test.txt', 'r')

numChars = 0
numLines = 0
numWords = 0

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.

#read the content of file 
data = fname.read()


#get the length of the data 
numChars = len(data)

# gets the number of lines 
CoList = data.split("\n")
for i in CoList: 
    if i:
        numLines += 1
        

# gets the number of words
x = data.split()
numWords = len(x)




# output code below is provided for you; you should not edit this
print('%d characters'%numChars)
print('%d lines'%numLines)
print('%d words'%numWords)

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
#INCLASS CODING 


# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your a2feed.txt file first
# Opens the file named test.txt
fname = open('/kaggle/input/a2data/test.txt', 'r')

numChars = 0
numLines = 0
numWords = 0

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.

#read the content of file 
# data = fname.read()

for line in fname: 
    words = line.split()
    numWords += len(words)
    
    line = line.rstrip() # Removes invisble characthers 
    numChars += 1
    
    numLines += 1



#get the length of the data 


# gets the number of lines 

        

# gets the number of words




# output code below is provided for you; you should not edit this
print('%d characters'%numChars)
print('%d lines'%numLines)
print('%d words'%numWords)

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/sherlock/sherlock.txt', 'r')

numChars = 0
numLines = 0
numWords = 0


# Add your code below

#read the content of file 
data = fname.read()



#get the length of the data 
numChars = len(data)

# gets the number of lines 
CoList = data.split("\n")
for i in CoList: 
    if i:
        numLines += 1
        
# gets the number of words
x = data.split()
numWords = len(x)

# output code below is provided for you; you should not edit this
print('%d characters'%numChars)
print('%d lines'%numLines)
print('%d words'%numWords)

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()

# Inclass Coding 
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/sherlock/sherlock.txt', 'r')

numChars = 0
numLines = 0
numWords = 0


# Add your code below


for line in fname: 
    words = line.split()
    numWords += len(words)
    
    line = line.rstrip() # Removes invisble characthers 
    numChars += 1
    
    numLines += 1


# output code below is provided for you; you should not edit this
print('%d characters'%numChars)
print('%d lines'%numLines)
print('%d words'%numWords)

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()