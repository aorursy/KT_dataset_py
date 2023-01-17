mylist = [1, 2, 3, 4, 5, 6]

#I can use the "for" and "in" commands to iterate through the items in the list. I will multiply each 

for eachitem in mylist:

    print(eachitem*eachitem)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

#I want to introduce a second variable that can get larger with each item. I'll call it collector.

#first I will zero out my new variable.

collector = 0

#now I will write the iteration loop.

for eachitem in mylist:

    #Each iteration will simply collect each item.

    collector = eachitem + collector

    #This print is not necessary but I want to see the collector growing as it loops through.

    print ("Collector has collected " + str(collector) + " collectors")



#After the loop is complete we will print the result.

print("\nAnd the final result is "+ str(collector))





#maybe you wanted me to call it total. But I wanted to call it collector. 

total = 0

s = "This is a test string for HCDE 530"

# Add your code below

#First I will use the split method on variable s to create a new list longstring.

longstring = s.split()



#Now I will make the loop that prints each word. I will use word as the holding variable for my loop.

for word in longstring:

    print(word)

# Add your code here

#I'm declaring the list again because it just makes it easier to debug.

mylist = [1, 2, 3, 4, 5, 6,4,8,7]



find = 1

replace = "four"

    

#I am going to nest the index function within declaring a new item of my list. This will go find 4 and replace it with "four." I wanted to do fancier things but not yet.

mylist[mylist.index(4)] = "four"

print(mylist)





# The code below allows you to access your Kaggle data files

import os

#This was useful and I don't need it so commenting it out.

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

for line in fname:

    print(line.strip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



#Note to instructor -- I think these instructions would be cleaer if you specified the INDEX NUMBER rather than the CHARACTER NUMBER since there is no such thing as a CHARACTER NUMBER in Python.



# Add your code below

#Iterate through the list and print only the breeds containing terrier

#Setup the for loop

for breed in dogList:

#Setup the first condition, searching for lower case t

    if "terrier" in breed:

#        print(breed) debugging line not needed for final output

        print(breed.find("terrier"))

    #Setup the second condition with a capitol T.

    elif "Terrier" in breed:

#       print(breed) debugging line not needed for final output

       print(breed.find("Terrier"))

    #This is the catch all for the rest. IT doesn't really matter what I put for the find argument. 

    else:

       print(breed.find("terrier"))
binList = [0,1,1,0,1,1,0]



# Add your code below



#Setup the foor loop

for x in binList:

    #Setup the print condition.

    if x == 1:

        print("One")

#Nothing else to setup since we are not doing anything with the other items.
#Setup the foor loop

for breed in dogList:

    #Including both Bulldog and bulldog, but the list doesn't contain any bulldog. 

    if "Bulldog" in breed or "bulldog" in breed:

        print(breed)

    
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/a2data/test.txt', 'r')





# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



#Setup the for loop.

for line in fname:

    #Split the line at default whitespace " " and declare this as a new list lineList

    lineList = line.split()

    

    #Get the length of the line and add it to the colelctor representing the number of characters.

    numChars = numChars + len(line)

    

    #Each time through the loop, add 1 to the value of numLines.

    numLines = numLines + 1

    

    #Get the length of the line list, representing the number of words in the list, and add to the collector for number of words.

    numWords = numWords + len(lineList)

   

#Debug code below.

#    print(lineList)

#    print(line.strip())

#    print("Line is " + str(len(line)) + " characters long")

#    print("Line has " + str(len(lineList)) + " words")



# output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
#Open the file

watson = open('/kaggle/input/sherlocktxt/sherlock.txt', 'r')



#Count the number of lines.

lineCount = 0

watsons = 0

for line in watson:

    lineCount = lineCount + 1



    #look for the term Watson.

    if "Watson" in line:

        watsons = watsons + 1

        

    #Print any line that has the word "clue" in it. We only want it if it's a lower case clue clue.

    if "clue" in line:

        print(line.strip())

        





print("\n\nLine count: " + str(lineCount))

print("Watsons: " + str(watsons))







#Close the file

fname.close()