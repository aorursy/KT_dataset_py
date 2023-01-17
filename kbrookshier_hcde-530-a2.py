mylist = [1, 2, 3, 4, 5, 6]



# Iterate over list, printing each item

for item in mylist:

    print(item)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



# Sum up the values in the list

for item in mylist:

    total += item



print(total)
s = "This is a test string for HCDE 530"

# Add your code below



# Create a list of the characters from the string

word_list = s.split()



# Print words from the list

for word in word_list:

    print(word)
# Add your code here

mylist[3] = "four"

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

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



# For any type of terrier, print the index of the 't'

for species in dogList:

    if "terrier" in species or "Terrier" in species:

        print(species.find('errier')-1)

    else:

        print(-1)
binList = [0,1,1,0,1,1,0]



# Add your code below



# Print one if the int value is 1

for current_item in binList:

    if current_item == 1:

        print('one')
# Print dog species from list that are bulldogs

for species in dogList:

    if 'bulldog' in species.lower():

        print(species)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words



# Updating the numChars, numLines, and numWords variables.

for line in fname:

    numChars += len(line)

    numLines += 1

    

    if ' ' in line:

        line = line.split(' ')

        numWords += len(line)

    else:

        numWords += 1





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

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words



# Updating the numChars, numLines, and numWords variables.

for line in fname:

    numChars += len(line)

    numLines += 1

    

    if ' ' in line:

        line = line.split(' ')

        numWords += len(line)

    else:

        numWords += 1



# output code below is provided for you; you should not edit this

print("The Sherlock text file contains:")

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()