mylist = [1, 2, 3, 4, 5, 6]



#In order to iterate each element in the list, I used a for loop calling for each number in the list to be iterated in a new line. 



for element in mylist:

    print(element)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



#To add total to the elements in mylist, I used a for loop function to call mylist, then in the same block I added the total value by using a +, adding it to the number of elements in the list. Then I printed total, which should be an additional of all the numbers, or 21.



total = 0



for element in mylist:

    total = total + element

    

print(total)
s = "This is a test string for HCDE 530"

# Add your code below



# First I created a for loop using the ".split" function which commands the code to print each words in "s" individually. I included that function in the for loop for word, in which each word should be divided up individually. Then in the next line, I printed "word", which should print each of the words in "s" in a seperate line. 

for word in s.split():

    print(word)
# Add your code here

#So I used indexing here by refering to my notes. Basically I used the index function to indicate that the element "4" should be replaced by the word "four" by using the equation symbol. By equating it to the word "four" in parenthesis, the function should print the mylist with the word four instead of the number 4. 

mylist [mylist.index(4)]="four"

print(mylist)
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below



#I did this by following along with the 2.1 tutorial on Canvas. I added the data set to my notebook, which activates the import code above. Then the fname funciton represents the txt files in the code. To print each line of the file in a seperate line, I used a for loop. To remove each extra space between lines, I added the ".strip" to the print function.  

for line in fname:

    print(line.strip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below



#This kinda tricky for me so I'm gonna go really slow. First I'm creating a for loop to iterate dogList.

for dog in dogList:



   #If the word that is being iterated on is "terrier", then the index will print the character number where the string "terrier" starts. 

    if "terrier" in dog: 

        print(dog.find("terrier"))

    

    #To account for instances of "Terrier" where the first letter is capatilized (I'm not too sure if the case will affect it, but just to be sure), I did the same thing here except using capital T.

    if "Terrier" in dog:

        print(dog.find("Terrier"))



     #If the line in the list does not include "terrier" or "Terrier", the code should print (-1).

    else:

        print(-1)
binList = [0,1,1,0,1,1,0]



# Add your code below



#I used a for loop to iterate the items in binList

for item in binList:

    

    #I used the "==" to indicate in the function that if the item is equal to 1, it will print the word "one"

    if item == 1:

        print("One")

        

    #Then, if any of the items in the binList are not equal to 1, the function should print nothing, which I indicated in this function as pass. 

    else:

        pass
# Here I created a for loop to iterate dogList from the previous section. 

for dog in dogList:



    #Im telling the code that if the word it's reading in the list is "bulldog", to print the whole name of the dog. 

    if "bulldog" in dog:

        print(dog)

    #here I'm doing the same thing but accounting for "Bulldog" with a capital B. 

    if "Bulldog" in dog:

        print(dog)

    #if the word does not include "bulldog" or "Bulldog", print nothing. 

    else:

        pass

numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#First I created a for loop function to iterate the lines in fname

for line in fname:

    

    #then I tell the function to split each of the lines into words by equating "words" to the split lines. 

    words = line.split()

    

    #then I use numChars to count the number of characters in the line 

    numChars = numChars + len(line)

    #I use numLines to count the numer of lines in the file

    numLines = numLines + 1

    #then I do the same thing as I did with numChars to numWords, to count the number of words in the lines. 

    numWords = numWords + len(words)

    



# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
#Following the same steps as the exercise above, I'm indentifying the variables used to represent the characters, lines, and words. 

#c for characters

cSherlock = 0

#l for lines

lSherlock = 0

#w for words

wSherlock = 0



#then I bring in the txt file and call it fname for simplicity

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



#First I created a for loop function to iterate the lines in fname

for line in fname:

    

    #then I tell the function to split each of the lines into words by equating "words" to split lines with .split. 

    words = line.split()

    

    #then I use cSherlock to count (#using len) the number of characters in the line 

    cSherlock = cSherlock + len(line)

    #I use lSherlock to count the numer of lines in the file

    lSherlock = lSherlock + 1

    #then I do the same thing as I did with cSherlock to wSherlock, to count the number of words in the lines. 

    wSherlock = wSherlock + len(words)



    

#I brought these print functions in from the last exercise, but edited the the variable names for characters, lines, and words. 

print(f'characters {cSherlock}')

print(f'lines {lSherlock}')

print(f'words {wSherlock}')



# Lastly, I close the file for good practice. Ta da!

fname.close()