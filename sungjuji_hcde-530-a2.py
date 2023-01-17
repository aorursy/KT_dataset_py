mylist = [1, 2, 3, 4, 5, 6]

for numbers in mylist:

    print(numbers)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0

for x in mylist:

    total = total + x

print(total)

s = "This is a test string for HCDE 530"

# Add your code below



for x in s.split():

    print(x)



# Add your code here



mylist[3] = "four"

mylist
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

# Add your code below

fname = open('/kaggle/input/a2data2/test.txt', 'r')

for line in fname:

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below







for dogtype in dogList:

    if "terrier" in dogtype:

        terrierindex = dogtype.find('terrier')

        print("substring 'terrier' found at index:", terrierindex)

    if "Terrier" in dogtype:

        Terrierindex = dogtype.find('Terrier')

        print("substring 'Terrier' found at index:", Terrierindex)    

    if "errier" not in dogtype:

        print("-1")

        









    

binList = [0,1,1,0,1,1,0]



# Add your code below



for num in binList:

    if num == 1:

        print("One")

    #else:

        #print("dont print")

for dogtype in dogList:

    if "bulldog" in dogtype:

        print(dogtype)

    if "Bulldog" in dogtype:

        print(dogtype)    

numChars = 0

numLines = 0

numWords = 0





# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data2/test.txt', 'r')





# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for line in fname:

    wordslist = line.split()

    numLines = numLines + 1

    numWords = numWords + len(wordslist)

    numChars = numChars + len(line)



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()

# Add your code below

fname = open('/kaggle/input/sherlock/sherlock.txt', 'r')



numChars = 0

numLines = 0

numWords = 0



for line in fname:

    wordslist = line.split()

    numLines = numLines + 1

    numWords = numWords + len(wordslist)

    numChars = numChars + len(line)



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)

    

fname.close()