mylist = [1, 2, 3, 4, 5, 6]

for number in mylist:

    print(number)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for number in mylist:

    total += number #update the total by adding each number in the list

print (total)

s = "This is a test string for HCDE 530"

# Add your code below

for word in s:

    cleaned = s.split() #split string into words

print (cleaned)

# Add your code here

mylist[3] = 'four'

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

#fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

data = open('/kaggle/input/sample/test.txt','r')



# It's good practice to close your file when you are finished. This is in the next line.

data.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]





# Add your code below

for dog in dogList:

    if "terrier" in dog: #check if 'terrier'is in the dog's name

        a = dog.find("terrier") #if yes, store index value of where 'terrier' starts

        print (a)

    elif "Terrier" in dog: #check if 'Terrier'is in the dog's name

        b = dog.find("Terrier") #if yes, store index value of where 'Terrier' starts

        print (b)

    else:

        print("-1")
binList = [0,1,1,0,1,1,0]



# Add your code below

for number in binList:

    if number == 1: #check if number is equal to one

        print("One")
for dog in dogList:

    if ("Bulldog" in dog) or ("bulldog" in dog): #accounts for both "Bulldog" or "bulldog"

        print(dog)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

data = open('/kaggle/input/sample/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in data:

    numLines += 1 #add one for each line read

    words = line.split() #split each line into words

    numWords += len(words) #count number of words in each line

    numChars += len(line) #count number of characters in each line



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

data.close()
# Add your code below

sher = open('/kaggle/input/sample2/sherlock.txt', 'r')

c = 0

w = 0

l = 0



for line in sher:

    l += 1 #count total number of lines

    words = line.split() #split each line into words

    w += len(words) #count number of words by counting the length of previous string above

    c += len(line) #count number of characters by counting the length of unsplit string

    

print('%d characters'%c)

print('%d words'%w)

print('%d lines'%l)



sher.close()