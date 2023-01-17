mylist = [1, 2, 3, 4, 5, 6]

for number in mylist:
    print(number)
    
# I made a variable called number, then printed each variable within the list

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0
for x in mylist:
    total = total + x
    
print (total)

# The total starts at 0, then increases by each value in the list and prints the end total
s = "This is a test string for HCDE 530"
# Add your code below


newList = s.split()  #This splits s into a list called newList

for x in newList:    #This iterates through newList and prints each word on its own line
    print(x)
    


# Add your code here 

# mylist[3] = "four"  # 3 is the 3rd element in the list, so I replaced this element with "four"

#print(mylist) 

for n, i in enumerate(mylist):
    if i == 4:
        mylist[n] = 'four'
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
    print(line.rstrip())

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below


#for allWords in dogList:
    
#    if "terrier" in allWords:
#        print(allWords.find("terrier"))   #This is saying, if you find "terrier" then print where you found it within the dogList
        
#    elif "Terrier" in allWords:
#        print(allWords.find("Terrier"))    #This is saying, if you find "Terrier" then print where you found it within the dogList
    
#    else:
#        print(-1)                            # If it doesnt find either, then just print -1
        
        
        
# From class        
for dog in dogList:
    result = dog.find("terrier")
    if result == -1:
        result = dog.find("Terrier")
    print(dog, result)
binList = [0,1,1,0,1,1,0]

# Add your code below

for allNum in binList:  
    
    if allNum == 1:
        print("One")   # for all numbers within the list, if its equal to 1, print "one"
#for allWords in dogList:
    
#    if allWords.find("Bulldog") != -1:
       # print(allWords)   #This is saying, if you find Bulldog (it doesnt evaluate to -1) then print the word
        
   # elif allWords.find("bulldog") != -1:
     #   print(allWords)    #This is saying, if you find bulldog (it doesnt evaluate to -1) then print the word
    
word =  "Bulldog"

#for dog in dogList:
 #   if dog.lower().find(word.lower()) != -1:
  #      print(dog)                          #look for these values, make sure theyre lowercase, if the dog in lowercase is found, then see what the result is
        
        
for dog in dogList:
    if word.lower() in dog.lower():
        print(dog)
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.

# for line in fname:
    
#    numLines = numLines + 1    # number of lines increases each time it reads a line in the list
    
#    wordsList = line.split()
#    numWords = numWords + len(wordsList) # Takes the length of each split line and adds it together
    
#    line = line.rstrip()
#    numChars = numChars + len(line)  # Takes the extra space off each line, then counts the length of each line and adds it together
    

    
for line in fname:
    words = line.split()
    numWords += len(words)
    
    line = line.rstrip()
    numChars += len(line)
    
    numLines += 1

# output code below is provided for you; you should not edit this

print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below

numChars = 0
numLines = 0
numWords = 0


f2name = open('/kaggle/input/sherlocktxt/sherlock.txt', 'r')


for line in f2name:
    
    numLines = numLines + 1    # number of lines increases each time it reads a line in the list
    
    wordsList = line.split()
    numWords = numWords + len(wordsList) # Takes the length of each split line and adds it together
    
    line = line.rstrip()
    numChars = numChars + len(line)  # Takes the extra space off each line, then counts the length of each line and adds it together

print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

fname.close()