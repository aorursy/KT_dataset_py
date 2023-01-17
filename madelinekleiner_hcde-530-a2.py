mylist = [1, 2, 3, 4, 5, 6]



# iterate through mylist and print each element

for num in mylist:

    print(num)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



# iterate through mylist and sum all elements

for num in mylist:

    total = total + num

print(total)

s = "This is a test string for HCDE 530"

# split string s into words on the whitespace

split_s = s.split()



# iterate through the split list and print each word on its own line

for word in split_s:

    print(word)
# replace the int 4 with the string "four" in mylist

mylist[3] = "four"        

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# print the file we opened and remove the newline character at the end of each line

for words in fname: 

    print(words.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# map function to convert dogList strings to lowercase

dogListChangeCase = map(lambda x:x.lower(), dogList)



# convert lowercase into a new dogList

dogListLowerCase = list(dogListChangeCase)



# iterate over breeds in dog list and print where "terrier" or "Terrier" starts, else print -1

for breed in dogListLowerCase:

    x = breed.find("terrier")

    print(x)

binList = [0,1,1,0,1,1,0]



# iterate over binList

for x in binList:

    

    # if item is 1 print "one"

    if x == 1:

        print("One")



# iterate over dogListLowerCase

for breed in dogListLowerCase:

    

    # find "bulldog" in the list

    x = breed.find("bulldog")

    

    # if "bulldog" is found, print the breed

    if x != -1:

        print(breed)

    
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

# print the file we opened and remove the newline character at the end of each line



# count the number of lines, characters, and words in the file

for lines in fname:

    numLines = numLines + 1

    numChars += len(lines)

    numWords += len(lines.split())



# print the file we opened and remove the newline character at the end of each line

for words in fname: 

    print(words.rstrip())



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# initialize counting variables

numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/sherlock/sherlock.txt', 'r')



# count the number of lines, characters, and words in the file

for lines in fname:

    numLines = numLines + 1

    numChars += len(lines)

    numWords += len(lines.split())



# output the number of characters, lines, and words

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# close the file

fname.close()