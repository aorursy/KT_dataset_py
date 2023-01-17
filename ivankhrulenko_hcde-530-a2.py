mylist = [1, 2, 3, 4, 5, 6]

for element in mylist:

    print(element)

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for element in mylist:

    total+=element

print(total)

s = "This is a test string for HCDE 530"

# Add your code below

for word in s.split():

    print(word)

    
# Add your code here

mylist[mylist.index(4)]="four"

print(mylist)
# The code below allows you to access your Kaggle data files

import os

#print(os.listdir("../input"))

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below

for line in fname:

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

#since I'll have to do it twice in this assignment, I am going define a function

def dogFind(hound, breed):

    #the function accepts a string to search in the list item (breed)

    #it also accepts a string to seach in (hound)

    #the function will take care of caps vs lower case

    #function returns the position of the breed in hound 

    #or -1 if breed is not found in hound

    dogStart = hound.find(breed.lower())

    #if breed is not found, let's try Breed. 

    if not (dogStart>=0):

        dogStart=dog.find(breed.title())

    return dogStart



for dog in dogList:

    #since the assignment does not specify which occurrence of t/Terier

    #we should be concerned abuut in "terrier Terrier" scenario,

    #let's assume we are looking for the first occurence only

    print (dogFind(dog, "terrier"))

binList = [0,1,1,0,1,1,0]



# Add your code below

#assuming that the list could contain only 0s or 1s

for num in binList:

    if num:

        print("One")

    
for dog in dogList:

    if dogFind(dog,"bulldog")>=0:

        print(dog)

        
# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your test.txt file first, in a new dataset called a2data

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

# Again, since we need to do it twice, I'll define a function

def fileStats (handle):

    # haven't read anything about other data structures yet

    # so the function will take file handle as an input

    # and return a list of counts from smaller to bigger category,

    # i.e. [<characters>, <words>, <lines>] 

    numChars = 0

    numLines = 0

    numWords = 0

    for line in handle:

        numChars+=len(line)

        numWords+=len(line.split())

        numLines+=1

    return [numChars, numWords, numLines]



#now let's feed the file to our function

results = fileStats(fname)

# output code below is provided for you; you should not edit this

print(f'characters {results[0]}')

print(f'lines {results[2]}')

print(f'words {results[1]}')



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below



#open the file

f = open('/kaggle/input/sherlock.txt', 'r')



#run stats

results = fileStats(f)



# should have wrapped this into a function too

print(f'characters {results[0]}')

print(f'lines {results[2]}')

print(f'words {results[1]}')



#close the file

f.close()