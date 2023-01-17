mylist = [1, 2, 3, 4, 5, 6]

for int in mylist:

        print (int)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for int in mylist:

    total = int + total

    print(total)



s = 'This is a test string for HCDE 530'

# Add your code below

word = s.split()

for w in word:

    print(w)





mylist[4]='four'

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below

for line in fname:

    print(line.strip())





# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



for dogtype in dogList:

    print(dogtype.find('terrier'))

    print(dogtype.find('Terrier'))

binList = [0,1,1,0,1,1,0]



# Add your code below

for num in binList:

    if num == 1:

        print ("One")

    else:

        pass

for dogtype in dogList:

    if 'Bulldog' in dogtype:

        print (dogtype)

    elif 'bulldog' in dogtype:

        print(dogtype)

    else: 

        pass
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/test.txt', 'r')



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

numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname2 = open('/kaggle/input/sherlock.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for line in fname2:

    wordslist = line.split()

    numLines = numLines + 1

    numWords = numWords + len(wordslist)

    numChars = numChars + len(line)





# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname2.close()