mylist = [1, 2, 3, 4, 5, 6]



#print each number in list

for number in mylist:

    print(number)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



#accumulate values in list

for number in mylist:

    total += number



#print sum of numbers in list

print(total)

s = "This is a test string for HCDE 530"



#print each word in string on seperate line

for word in s.split():

    print(word)

# Add your code here



#find and replace 4 with four in array

for value in mylist:

    if mylist[value-1] == 4:

        mylist[value-1] = "four"



#print value of list

print(mylist)

# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below



#print content of file

for line in fname:

    print(line.rstrip())

# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



#loop through lists

for breed in dogList:

    #check to see if text has "Terrier" and print character number where Terrier starts

    if breed.find("Terrier") >= 0:

        print(breed.find("Terrier"))

        

    #check to see if text has "terrier", print character number where terrier starts

    elif breed.find("terrier") >= 0:

        print(breed.find("terrier"))

        

    #print "-1" when neither Terrier or terrier is found in text

    else:

        print("-1")

        
binList = [0,1,1,0,1,1,0]



# Add your code below



#loop through each value in list

for item in binList:

    #when value is "1", print "One"

    if item == 1:

        print("One")

#loop through each value in list

for breed in dogList:

    #when value has Bulldog or bulldog, print the value

    if breed.find("Bulldog") >= 0 or breed.find("bulldog") >= 0:

        print(breed)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#loop through each line in file

for line in fname:

    #increase count for line

    numLines += 1

    

    #loop through each word and increase word count

    for word in line.split():

        numWords += 1

        

    #loop through each character in line and increase character count

    for chars in line:

        numChars += 1



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below



#open sherlock.txt

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



#loop through each line in file

for line in fname:

    #increase count for line

    numLines += 1

    

    #loop through each word and increase word count

    for word in line.split():

        numWords += 1

        

    #loop through each character in line and increase character count

    for chars in line:

        numChars += 1

        

# print number of characters, lines and words in file

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



#close file

fname.close()