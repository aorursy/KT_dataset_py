mylist = [1, 2, 3, 4, 5, 6]

for it in mylist: 

    print(it)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:

    total = total + x

print (total)
s = "This is a test string for HCDE 530"

# Add your code below

s.split()

for ss in s.split():

    print(ss)

# Add your code here

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

for line in fname:

    print(line.strip())





# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



for x in dogList:

    print(x.find('ter'or 'Ter'))

binList = [0,1,1,0,1,1,0]

for x in binList:

    if x == 1:

        print('One')
for x in dogList:

    if (x.find('bulldog' or 'Bulldog') != -1):

        print(x)

        

#I tried to run this code and it didn't work. I ran it in the terminal and it worked fine so not sure what's going on here!
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data



fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    line = line.strip("\n")

    words = line.split()

    numLines += 1

    numWords += len(words)

    numChars += len(line)





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



fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



for line in fname:

    line = line.strip("\n")

    words = line.split()

    numLines += 1

    numWords += len(words)

    numChars += len(line)

    

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')

    

fname.close()


