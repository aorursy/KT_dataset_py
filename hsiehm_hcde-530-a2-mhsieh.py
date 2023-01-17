mylist = [1, 2, 3, 4, 5, 6] # list that will be iterated

for item in mylist:

    print(item) # print automatically includes a new line character, so we can just print the item

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0 # assign a variable to keep track of the total

for x in mylist: # use the list from before

    total = total + x  # add the iteration to the current total

print(total) # print out the total after the for loops has finished running

s = "This is a test string for HCDE 530"

# Add your code below

s2 = s.split() # use the Python defined split function for strings to create a new list, which defaults to using whitespace as the splitter

for x in s2: # iterate through the new list

    print(x) # print out each indexed word in the list

# Add your code here

for x in mylist: # iterate through the list

    if(x == 4): # check to see if the iteration matches 4

        mylist[x-1] = "four" # assign the string if we met our previous condition; we need to subtract 1 to match the index.

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below

for x in fname: # iterate through our file, looking at each line

    words = x.strip("\n") # remove the new lines in the file, since print includes new lines

    print(words) # print out the words as they appear in the file



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for x in dogList:

    process = x.title() # change every word to Title case so we only need to perform one find()

    charNum = process.find("Terrier") # check for the existence of string terrier or Terrier

    print(charNum) # print out -1 if terrier is not found, otherwise the index - 1 where terrier starts in the line

       

binList = [0,1,1,0,1,1,0]



# Add your code below

for x in binList:

    if(x==1): # check if the iterated value is actually 1

        print("One") # print string One

index = 0 # keep track of our iterations

for x in dogList:

    process = x.title() # change every word to Title case so we only need to perform one find()

    if(process.find("Bulldog") != -1): # check for Bulldog

        print(dogList[index]) # print the line if we were able to find Bulldog in the line; use original dogList so we're not altering the original data

    index = index + 1 # increment
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for x in fname:

    numLines = numLines + 1 # each iteration is a new line

    numWords = numWords + len(x.split()) # the number of splits by default whitespace is a word in a line

    for y in x: # each word can be turned into a list

        numChars = numChars + len(y) # add the length of each word to the total number of characters





# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below

def counter(filename): # define a method for getting a count for lines, words, and chars in a file.

    numChars = 0 # set counter variable for number of characters

    numLines = 0 # set counter variable for number of lines

    numWords = 0 # set counter variable for number of words

    f = open(filename, 'r') # open the filename specified for reading



    for x in f: 

        numLines = numLines + 1 # each iteration is a new line

        numWords = numWords + len(x.split()) # the number of splits by default whitespace is a word in a line

        for y in x: # each word can be turned into a list

            numChars = numChars + len(y) # add the length of each word to the total number of characters

    

    # print out values at the end of counting the file

    print(f'characters {numChars}')

    print(f'lines {numLines}')

    print(f'words {numWords}')

    

    f.close() # close our file since we have finished using it



counter('/kaggle/input/sherlock.txt') # run the method using our data file