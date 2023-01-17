mylist = [1, 2, 3, 4, 5, 6]

for item in mylist:

    print(item)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for item in mylist:

    total += item

    

print(total)

s = "This is a test string for HCDE 530"

# Add your code below



# iterate through the splited tokens

for word in s.split():

    print(word)

    
# Add your code here



# find the index of element '4' assuming its existence

# then use the index to update that value to string var four

mylist[mylist.index(4)] = 'four'

print(mylist)

# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/hcde530a2data/test.txt', 'r')



# Add your code below

for line in fname:

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for dog in dogList:

# find() returns -1 when it does not exist, thus the max() would get the index if it exists

# could also use lower() here 

    print(max(dog.find('terrier'), dog.find('Terrier')))

    

    
binList = [0,1,1,0,1,1,0]



# Add your code below



for item in binList:

    if item == 1:

        print("One")

for dog in dogList:

# convert the string to lowercase then compared with 'bulldog' for the case-insensitive req.

    if dog.lower().find('bulldog') > -1:

        print(dog)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/hcde530a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for line in fname:

    numLines += 1

    numWords += len(line.split())

    numChars += len(line)





# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below



num_of_chars = 0

num_of_lines = 0

num_of_word = 0



f = open('/kaggle/input/hcde530a2datasherlock/sherlock.txt', 'r')



for line in f:

    num_of_lines += 1

    num_of_word += len(line.split())

    num_of_chars += len(line)



print(f'{num_of_chars} num_of_chars')

print(f'{num_of_lines} num_of_lines')

print(f'{num_of_word} num_of_word')



f.close()
