mylist = [1, 2, 3, 4, 5, 6]

#for loop to print each element in mylist on a new line
for x in mylist:
    print(x)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0

# for loop that updates total in each iteration by adding the next number in mylist
for y in mylist:
    total = total + y
    
print(total)

s = "This is a test string for HCDE 530"
# Add your code below

#split the string into words so that the for loop can understand it.
words = s.split()

#use for loop to write words on new line.
for line in words:
    print(line)
        

    
        
    

# Add your code here

#replace mylist[3] which is 4 to the string 'four'
mylist[3] = 'four'
print(mylist)




# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below
# we are running a for loop to print each line of text in the file. Used rstrip() to remove new the extra newline characters.
for line in fname:
    print(line.rstrip())

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

#create for loop to iterate in dogList
for dog in dogList:
    
    #if the word terrier is in the current word being iterated on, find() will output the index of where 'terrier' starts.
    if 'terrier'in dog:
        print(dog.find('terrier'))
        
    #same idea here
    if 'Terrier' in dog:
        print(dog.find('Terrier')) 
        
    #for everything else that does not include terrier or Terrier, we print -1    
    else:
        print('-1')
        
        

binList = [0,1,1,0,1,1,0]

# Add your code below

#iterating through binlist via for loop
for i in binList:
    
    #if the element is 1, then print 'one'
    if i == 1:
        print('one')
        
        #if not 1, then don't print anything. I assume this means a blank space or a next line. 
    else:
        print(' ')
#iterate through dogList
for d in dogList:
    
    #if the term Bulldog exists in d, print the full term.
    if 'Bulldog' in d:
        print(d)
        
    #accounting for lowercase
    elif 'bulldog' in d:
        print(d)
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.

#for loop to iterate in file
for chars in fname:
    
    #add rstrip method to chars to remove unnecessary invisible spaces. Note that with numChars this goes from 22 to 20 with rstrip.
    chars = chars.rstrip()
    #define c as the length of characters in the first line
    c = len(chars)
    
    #add the length of characters to numChars after each line.
    numChars = numChars + c

    
# for some reason I have to keep opening the file to run the next loop.
fname = open('/kaggle/input/a2data/test.txt', 'r')

#create for loop to iterate in file. 
for line in fname:
    #add 1 to numLines each iteration.
    numLines = numLines + 1
    
    
#open file again
fname = open('/kaggle/input/a2data/test.txt', 'r')

# for loop to iterate.
for w in fname:   
    
    #split string into words
    words = w.split()
    
    #wordline is the number of words per line
    wordline = len(words)
    
    #add words per line to the number of words after each iteration.
    numWords = numWords + wordline


# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below

#open sherlock.txt
fn = open('/kaggle/input/a2data/sherlock.txt', 'r')


#define variables to start with 0. We will use this later
numChars = 0
numLines = 0
numWords = 0


#for loop to iterate in file
for chars in fn:
    
    #define c as the length of characters in the first line
    c = len(chars)
    
    #add the length of characters to numChars after each line.
    numChars = numChars + c

    
# for some reason I have to keep opening the file to run the next loop.
fn = open('/kaggle/input/a2data/sherlock.txt', 'r')

#create for loop to iterate in file. 
for line in fn:
    #add 1 to numLines each iteration.
    numLines = numLines + 1
    
    
#open file again
fn = open('/kaggle/input/a2data/sherlock.txt', 'r')

# for loop to iterate.
for w in fn:   
    
    #split string into words
    words = w.split()
    
    #wordline is the number of words per line
    wordline = len(words)
    
    #add words per line to the number of words after each iteration.
    numWords = numWords + wordline

    
# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()