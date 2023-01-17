mylist = [1, 2, 3, 4, 5, 6]

#declare a var called "nums"

for num in mylist:

    #values in nums will iterate over mylist and print one at a time

    print(num)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:

    # x = mylist[0] =1, total = 0 + 1 = 1

    # x = mylist[1] =2, total = 1 + 2 = 3

    #......

    total = total + x

print(total)
s = "This is a test string for HCDE 530"

# Add your code below

words = s.split()



#the value of words becomes: ['This', 'is', 'a', 'test', 'string', 'for', 'HCDE', '530']

#loop throug the words and print elements one by one



for temp in words:

    print(temp)
# Add your code here



# I declare a new list here so that I won't mess up the the orginal mylist

# mylist = [1, 2, 3, 4, 5, 6]



#deep copy

newlist = mylist[:]



newlist[newlist.index(4)] = "four"



print("this is the original list",mylist)



print("this is a new list", newlist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below

for line in fname:

    nline = line.rstrip()

    print(nline)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



#check each elements in dogList

for i in dogList:

    #I use the condition here since I don't know how to say find("terrier" or "Terrier")

    result = i.find('terrier')

    if result == -1:

        result = i.find('Terrier')

    print(i, result)
binList = [0,1,1,0,1,1,0]



# Add your code below



for i in binList:

    if i == 1:

        print("One")

        

#Am I understand this task correctly?
# I use var i as index here



numList = len(dogList)



#check each elements in dogList

for i in range (numList):

    if "Bulldog" in dogList[i] or "bulldog" in dogList[i]:

        print (dogList[i])
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for aLine in fname:



    #cound how many lines

    numLines = numLines + 1

    

    #count how many chars in each line

    numChars = numChars + len(aLine)

    

    #count how many words

    aWords = aLine.split()

    numWords = numWords + len(aWords)

    





# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
sChars = 0

sLines = 0

sWords = 0



fnameSherlock = open('/kaggle/input/sherlock.txt', 'r')





for sherlockLines in fnameSherlock:



    #cound how many lines

    sLines = sLines + 1

    

    #count how many chars in each line

    sChars = sChars + len(sherlockLines)

    

    #count how many words

    sherlockWords = sherlockLines.split()

    sWords = sWords + len(sherlockWords)

    



print('%d characters'%sChars)

print('%d lines'%sLines)

print('%d words'%sWords)



fnameSherlock.close()