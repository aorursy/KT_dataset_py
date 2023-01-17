mylist = [1, 2, 3, 4, 5, 6] # Create a variable called mylist. Assign it to a list of numbers between and including 1 through 6.
for num in mylist: # Create a for loop. For each num in mylist...
    print(num)#...print the current int of num in mylist.
total = 0 # start with a variable named total that keeps count of a desired input
nums = eval(input(' ')) # create another variable named nums that takes an input from the user in the form of a bracketed list. Then use the eval() function to calculate the user-input list. In this case, it's [1,2,3,4,5,6].
for x in nums: # declare a for loop in which for each variable x in the list nums...
    total = total + x #...evaluate and add to the total. Continue to do this until the list is complete.
print(total) # then, print the total. In this case, it should be 21.
s = "This is a test string for HCDE 530"
# Add your code below
words = s.split() #call the variable s, which is assigned to a string, then split the string into a list with the .split() function

for i in words:
    print(i)
# Add your code here
#mylist[3] = 'four' # assign the fourth item in mylist to the string 'four'
#print(mylist) # print the fourth item in mylist

#OR

for n, i in enumerate(mylist):
    if i == 4:
        mylist[n] = 'four'
print(mylist)
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input/a2data'): # changed path to a2data
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/test.txt', 'r') ## create a variable called fname that will open the file from our folder. Include the path to the file as a string.

# Add your code below
for line in fname: # create a for loop that iterates through each line within the file being opened
    print(line.strip()) # print each line and use the strip() function to remove the \n characters

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below
for dog in dogList:
    #if we found terrier then we know the character number where that starts
    result = dog.find('terrier')
    if result == -1:
        result = dog.find('Terrier')
    print(dog, result)
        
binList = [0,1,1,0,1,1,0]

# Add your code below
for item in binList:
    if item == 1:
        print("One")

word = "Bulldog"

#for dog in dogList: # use a conditional with the find() and lower() statements
#    if dog.lower().find(word.lower()) != -1:
#        print(dog)

# OR
for dog in dogList:
    if word.lower() in dog.lower():
        print(dog)
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('../input/a2data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
    
    words = line.split()
    numWords += len(words)
    
    line = line.strip()
    numChars += len(line)
    
    numLines += 1
# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below
sherlockChars = 0
sherlockLines = 0
sherlockWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('../input/sherlock/sherlock.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
    
    words = line.split()
    sherlockWords += len(words)
    
    line = line.strip()
    sherlockChars += len(line)
    
    sherlockLines += 1
# output code below is provided for you; you should not edit this
print(f'characters {sherlockChars}')
print(f'lines {sherlockLines}')
print(f'words {sherlockWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()