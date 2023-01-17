mylist = [1, 2, 3, 4, 5, 6]

for item in mylist:

        print(item)

        



    



    

    

    

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for item in mylist:

    total = total + item

print(total)



s = "This is a test string for HCDE 530"

# Add your code below

s_split = s.split(" ")

for item in s_split:

    print(item)

# Add your code here

mylist.insert(3,"four")

mylist.remove(4)

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/testtxt/test.txt', 'r')



# Add your code below





for line in fname:

    line = line.strip()

    print(line)





# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for item in dogList:

    if "Terrier" in item or "terrier" in item:

        print(len(item)-7)

    else: 

        print("-1")

        

binList = [0,1,1,0,1,1,0]



# Add your code below

for item in binList:

    if item == 1:

        print("One")

for item in dogList:

    if "Bulldog" in item or "bulldog" in item:

        print(item)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/testtxt/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.





for line in fname:

    list = line.split()

    line = line.replace("\n","")

    numLines = numLines + 1

    numWords = numWords + len(list)

    numChars = (numChars + len(line))

    



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



f = open('/kaggle/input/sherlocktxt/sherlock.txt', 'r')



for line in f:

    list = line.split()

    line = line.replace("\n","").replace("\r","")

    numLines = numLines + 1

    numWords = numWords + len(list)

    numChars = (numChars + len(line))

    

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



f.close()