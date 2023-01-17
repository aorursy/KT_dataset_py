mylist = [1, 2, 3, 4, 5, 6]

for number in mylist:

    print(number)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for num in mylist:

  total += num

print(total)



s = "This is a test string for HCDE 530"

for words in s.split():

    print (words)

for num in mylist:

    if num == 4:

      print("Four")

    else:

      print(num)

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

for dog in range(len(dogList)):

  if dogList[dog].find("terrier") >= 0:

    print(dog)

  elif dogList[dog].find("Terrier") >= 0:

    print(dog)

  else:

    print("-1")

binList = [0,1,1,0,1,1,0]

for num in binList:

 if num == 1:

    print("One")

for dog in range(len(dogList)):

  if dogList[dog].find("Bulldog") >= 0:

    print(dog)

  elif dogList[dog].find("bulldog") >= 0:

    print(dog)
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