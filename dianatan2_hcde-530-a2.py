mylist = [1, 2, 3, 4, 5, 6]

for x in mylist:

    print (x)

#I followed the instructions in the Week 3 lecture, assigned x as the varname and then printed the list as a loop.
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:

    total = total + x

print(total)



#I attempted to recreate this from the accumulation pattern in the lecture and first included the eval input line in the lecture. However, I realized that the eval input line looped the string that included the commentary on the line so I just removed it and the sum came out just by using total on mylist.
s = "This is a test string for HCDE 530"

# Add your code below

for x in s:

    words=s.split( )

    print (words[0])

    print (words[1])

    print (words[2])

    print (words[3])

    print (words[4])

    print (words[5])

    print (words[6])

    print (words[7])

# Add your code here

mylist[3]= 'four'

print(mylist)



#I indexed to where 4 is in the string and then printed the string with 4 as the replacement.
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt','r')



# Add your code below

for line in fname:

    print(line.strip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()



#I followed the instructions in the how to video with Brock and was initially successful in creating this in another notebook during the exercise. However, when I tried to access the same file here with the identical code, I received an error. 

#I read in the forum that closing and reopening this notebook would make this work so I just did that and kept the code the same and it worked.Then, I used the strip function to remove extra lines.
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for x in dogList:

    if  (x.find('terrier') != -1):

      print(7)

    else:

      print(-1)



#I used a for statement to break out the list in dogList and then I used find to find the word terrier. I realized that the capitalization of terrier and Terrier doesn't make a difference.

#If substring exists inside the string, it returns the index of first occurence of the substring.

#If substring doesn't exist inside the string, it returns -1.

#Then I did an if/else statement to make sure we only print the 7 if terrier is in the line and -1 if it is not. 
binList = [0,1,1,0,1,1,0]



# Add your code below

for x in binList:

    if  (x==1):

      print("One")

    else:

      print()



#I didn't know what "don't print anything" meant..I'm not sure if you wanted me to remove the extra lines but I just assumed we will keep the lines as is and only print out the lines with One.
for x in dogList:

    if  (x.find('Bulldog') != -1):

      print(x)

    else:

      print()

#Like the previous question, I assumed we wanted blank lines for the items that were not bulldog. I basically repeated the same process as the previous question, only to print x instead of the integer.
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

data=fname.read()

numChars=len(data)

datalist = data.split() 

numWords=len(datalist)

dataline= data.split("\n")

for x in dataline: 

    if x: 

        numLines += 1



# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()



#I started first by reading the file. Then I counted the characters with just the len function on the file.

#Then I split the line via words with the separation of spaces.

#Then I split the line further and counted the number of lines with the \n function.
# Add your code below

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

fname= open('/kaggle/input/sherlock/sherlock.txt','r')



numChars = 0

numLines = 0

numWords = 0



data=fname.read()

numChars=len(data)

datalist = data.split() 

numWords=len(datalist)

dataline= data.split("\n")

for x in dataline: 

    if x: 

        numLines += 1



# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()



#I inputed the file and changed the directory to reflect the Sherlock Txt file.

#Then I used the same code as the previous question.

#I'm not sure if I was supposed to print the directory but I just did it anyways since it helped me confirm that the file could be opened.