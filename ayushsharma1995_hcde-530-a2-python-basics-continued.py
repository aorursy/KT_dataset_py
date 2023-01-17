mylist = [1, 2, 3, 4, 5, 6]

for number in mylist:

    print(number)

#     print(f'{number}\n')
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for num in mylist:

    total += num

print(total)
s = "This is a test string for HCDE 530"

# Add your code below

newStr = s.split()

for word in newStr:

    print(word)
# Add your code here

for element in range(len(mylist)):

    if mylist[element] == 4:

        mylist[element] = 'four'

    else:

        continue



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

    line = line.rstrip()

    print(line)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

for item in dogList:

#Shorter method to avoid case sensitivity

#     item = item.casefold()

#     print(item.find('terrier'))

    if item.find("Terrier") > -1:

        print(item.find("Terrier"))

    elif item.find("terrier") > -1:

        print(item.find("terrier"))

    else:

        print('-1')
binList = [0,1,1,0,1,1,0]



# Add your code below

for number in binList:

    if number == 1:

        print("One")

    else:

        continue
bulldogList = []

for item in dogList:

    if item.find("Bulldog") > -1:

        bulldogList.append(item)

    elif item.find("bulldog") > -1:

        bulldogList.append(item)

    else:

        continue

print(bulldogList)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    #Remove additional new lines

    line = line.rstrip()

    #Every time the loop runs, increment the line count

    numLines += 1

    #Split the line based on whitespace to count words

    wordList = line.split()

    numWords += len(wordList)

    #For each line, iterate over it to count the characters (including whitespace between words)

    for letter in line:

        numChars += 1 #excluding new lines, but including whitespaces

#   To exclude the whitespace in character count as well, iterate over the newly split word list instead of the line in file

#     for letter in wordList:

#         numChars += len(letter)



# output code below is provided for you; you should not edit this

print(f'characters {numChars} [Excluding New Lines]')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
charCount = 0

wordCount = 0

lineCount = 0

# Add your code below

file2 = open("/kaggle/input/a2data2/sherlock.txt")

# contents = file2.read()

# print(len(contents.split()))



for line in file2:

    #Remove additional new lines

    line = line.rstrip()

    #Every time the loop runs, increment the line count

    lineCount += 1

    #Split the line based on whitespace to count words

    wordList = line.split()

    wordCount += len(wordList)

    #For each line, iterate over it to count the characters (including whitespace between words)

    for letter in line:

        charCount += 1

#   To exclude the whitespace in character count as well, iterate over the newly split word list instead of the line in file

#     for letter in wordList:

#         charCount += len(letter)



print(f'characters {charCount}')

print(f'lines {lineCount}')

print(f'words {wordCount}')



file2.close()