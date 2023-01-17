#I was honestly confused as to what to change about this code. 

mylist = [1, 2, 3, 4, 5, 6]



for x in mylist:

    print(x)

#Total creates a baseline of 0 and then the for loop adds each element to this 0 until a sum is reached

total = 0

for x in mylist:

    total = total + x

print(total)



#The string is split into separate pieces using the split function and each is printed on a new line due to the for loop



s = "This is a test string for HCDE 530"



s.split()



for x in s.split():

    print(x)

#This class is my first experience doing any programming work so I wasn't sure if there was a faster or more efficient way to do this.

#I deleted element 3 using the delete function and then inserted "four" into the position that 3 was in previously.



del(mylist[3])

mylist.insert(3,'four')



print(mylist)

# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below



#I stripped the new lines off using rstrip but I wasn't sure how to remove the file location at the top of the code.



for line in fname:



    x = line.rstrip()



    print(x)

fname.close()



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()


dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



#I worked on this question for over an hour and am completely stuck, my thought process was to use "find" like recommended in the question but it won't allow me to do so with a list.

#I understand the functionality of "find" to get the character number where "terrier" starts.

terrier = "Boston Terrier"



terrier.find("Terrier")



#But I have no idea how to use it in the context of a for loop involving a list and I can't find a section about "find" or some workaround in the lecture slides

#I will definitely want to go over this in the programming lab after next class since I was completely stumped. 



dogList.find("Terrier")
binList = [0,1,1,0,1,1,0]



# Add your code below



#The for loop took the list and put each number on a new line, then the boolean operation was able to separate the list.

#Once the list was separated, I printed "One" for each 1, and "" for each 0 in the list.



for x in binList:

   if x == 1:

        print("One")

   else:

        print("")

#I still had no idea how to properly use the find method 

#So I identified which string I was interested in using the breed variable and then I separated out the list using boolean operations



data = dogList

breed = "Bulldog"

filtering = [string for string in data if breed in string]

print(filtering)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#Number of Characters

#The read function reads the entire file and the length function counts the total number of characters

data = fname.read() 

numChars = len(data)



#Number of Words

#The split function separates the words into a list and the length function counts the number of items (3)

words = data.split()

numWords = (len(words))



#Number of Lines

#This one I admittedly had to do some research on since I was stumped, I found a function called "readlines" that reads all lines at once and returns each line as a string element.

#Once I found this function, I simply used the length function to count the number of strings received from reading the lines

numLines = len(open('/kaggle/input/a2data/test.txt').readlines())



# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below