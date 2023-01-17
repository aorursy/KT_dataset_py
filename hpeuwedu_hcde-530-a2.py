mylist = [1, 2, 3, 4, 5, 6]

#def variable to iterate over in mylist

for a in mylist:

#print each element on a new line

    print(a)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



# defining total as 0 because this value is able to be added to each element in mylist.

total = 0

# using for loop to iterate. "a" will pass through each element in mylist and then be added to "a" 

for a in mylist:

    #this basically means total = 0 + element in mylist everytime "a" variable passes through each index of mylist

    total = total + a 

#prints the sum of mylist which is "total"

print(total)



s = ["This is a test string for HCDE 530"]

# Add your code below

#using for loop because it will allow iteration through the entire string. "strg"is a variable. 

for strg in s: 

    # need to define a new object (not sure if this is called an object?) in order to use split function to split the string into a list of strings

    words = strg.split()

    # split() turned s to s = ["This","is","a","test", "string", "for", "HCDE", "530"] 

    #so using for loop to iterate through this new list using new variable "astrg" which will now loop through the new "s"

    for astrg in words:

    #prints each element in "words" on a new line

     print(astrg)

# Add your code here



#for loop to iterate through mylist

for a in mylist:

    #this defining a specific position in the index.In mylist at index 3 it now refers to "four"

    mylist[3] = "four"

    #prints mylist but because we defined index 3 to reference "four" it will replace 4 with "four"

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

#for loop to iterate through lines of fname. Words is a variable that will be passed through each index of fname

for words in fname:

    #prints each element word for word in fname. 

    print(words.strip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below  

#for loop to iterate through dogList

for dog in dogList:

    #when dog goes through dogList and finds character of terrier go to dlist

    dlist = dog.find("terrier")

    #if it returns negative 

    if dlist == -1: 

        #find "terrier" 

        dlist = dog.find("Terrier")

        #print character number 

        print(dlist,dog)

   

    

        

    

    



    
binList = [0,1,1,0,1,1,0]



# Add your code below

#for loop to iterate on list

for a in binList:

    #if else to print "one" if "a" is equal to 1 

    if a == 1:

        #print word one when "a" iterates through a 1

        print("One")

  

        

    

#defining variable 

a = "Bulldog"



#for loop to iterate through dogList

for dog in dogList: 

    #if dog doesn't equal to -1 find "a" which is referencing "Bulldog"

    if dog.lower().find(a.lower()) != -1:

        #print type of dog 

        print(dog)





    
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for char in fname:

    words = char.split()

    numWords += len(words)

    char = char.rstrip()

    numChars += len(char)

    numLines += 1

    

# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below

#defining variables

sherChars = 0 

sherLines = 0 

sherWords = 0 



#upload and open file to read

fname = open('/kaggle/input/sherlock/sherlock.txt', 'r')

#iterating on file

for words in fname:

    lines = words.split()

    sherLines += len(lines)

    words = words.rstrip()

    sherChars += len(words)

    sherWords += 1



print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



fname.close()


