mylist = [1, 2, 3, 4, 5, 6]



for x in mylist:

    print(x)

    

#The variable x is used to loop through mylist until every integer has been printed

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



for t in mylist:

    total = total + t

print(total)



#this code uses the variable t to loop through mylist, each integer is added to the variable total until they have all been added together. 
s = "This is a test string for HCDE 530"

# Add your code below



q = s.split()



for y in q:

 word = y.split()

 print(word)



#The variable y is used to loop through the string that was split using .split. 

#Every word is then printed on a seperate line. 

# Add your code here



mylist[4] = 'four'

print(mylist)



#using indexing the number 4 is pulled out and replaced with a string 

# The code below allows you to access your Kaggle data files



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data1/test.txt', 'r')





# Add your code below



for u in fname: 

    print(u.rstrip())

    

# printing the file and removing extra characters  

# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



for d in dogList: 

    location = d.find("terrier")

    if location == -1:

        location = d.find("Terrier")

    print(d, location)

   

# the list searches for "terrier, if not found moves on to "Terrier", if found prints the location. It loops for every item in list. 
binList = [0,1,1,0,1,1,0]



# Add your code below



for r in binList:

    if r == 1:

        print("One")





# I am using variable r to interate over the list binList. 

#If the item in the list equal 1 then it prints "one", if not it is skipped over. 













for dog in dogList: 

    if dog.find("Bulldog") != -1:

        print(dog)



   

#The for loops, loops through every element of dogList looking for bulldog. if found  (which means the value would not be -1) 

#it prints the values, if not it is skipped. 
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/a2data1/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for num in fname: 

    word = num.split()

    numWords = numWords + len(word)

    

    char = num.rstrip()

    numChars =  numChars + len(char)

    

    numLines =  numLines + 1



#IT loops through the file, first counting the number of words

# it then removes extra characters and looks for the number of characters excluding spaces  

#It then counts the number of lines  





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



# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data1/sherlock.txt', 'r')



for num in fname: 

    word = num.split()

    numWords =  numWords + len(word)

    

    char = num.rstrip()

    numChars = numChars + len(char)

    

    numLines =  numLines + 1



#It loops through the file, first counting the number of words

# it then removes extra characters and looks for the number of characters excluding spaces  

#It then counts the number of lines  





# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()