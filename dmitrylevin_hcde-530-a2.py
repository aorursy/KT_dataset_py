mylist = [1, 2, 3, 4, 5, 6]

for x in mylist:

  print(x)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:

    total = total + x

    print(x)
s = "This is a test string for HCDE 530"

# Add your code below

text = s.split()

for x in text:

  print(x)
# Add your code here

for item in mylist:

    if item == 4:

        item = 'four'

    print(item)

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



for item in dogList:

    index = item.find('terrier')

    if index == -1:

        index = item.find("Terrier")

    print(index)

       

   

# Add your code below

binList = [0,1,1,0,1,1,0]



# Add your code below



for item in binList:

    if item == 1:

        print("One")

        


for item in dogList:

    if item.find("Bulldog") > -1 or item.find("bulldog") > -1:

        print(item)

        
numChars = 0

numLines = 0

numWords = 0





# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below



for line in fname:

    numChars = numChars + len(line)

    numLines = numLines + 1

    numWords = numWords + len(line.split())

    

    



    

    

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

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



# Add your code below



for line in fname:

    numChars = numChars + len(line)

    numLines = numLines + 1

    numWords = numWords + len(line.split())

    

    



    

    

# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()