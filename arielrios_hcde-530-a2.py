mylist = [1, 2, 3, 4, 5, 6]

for x in mylist:

    print (x)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0

for x in mylist:

    total = total + x



print(total)

s = "This is a test string for HCDE 530"

# Add your code below



s = s.split()

curr = 0

for x in s:

    print(s[curr])

    curr = curr+1



# I feel that this could have been executed better
# Add your code here

curr = 0



while curr < len(mylist):

    if mylist[curr] == 4:

        mylist[curr] = "four"

        curr += 1

    else:

        curr += 1
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
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for dog in dogList:

    if dogList.find('terrier') or dogList.find('Terrier'):

        print(dog)

    
binList = [0,1,1,0,1,1,0]



# Add your code below

for num in binList:

    if num == 1:

        print("one")
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.







# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below