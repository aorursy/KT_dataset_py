mylist = [1, 2, 3, 4, 5, 6]
for elem in mylist:
        print(elem)
   

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.
total = 0
for x in mylist:
    total = total + x
print(total)


s = "This is a test string for HCDE 530"
# Add your code below
for x in s:
    x = s.split()
    print(x)
s = "This is a test string for HCDE 530"
words = s.split()
for i in words:
    print(i)
# Add your code here
del(mylist[3])
mylist.insert(3, 'four')
print(mylist)
mylist = [1, 2, 3, 4, 5, 6]
for n, i in enumerate(mylist):
    if i == 4:
        mylist[n] = 'four'
        print(n)
print(mylist)
# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below

for line in fname:
    print(line)
    
# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
for line in fname:
    print(line.rstrip())
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below
dogList.find('terrier')


dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]
for dog in dogList: 
    result = dog.find('terrier')
    if result == -1:
        result = dog.find('Terrier')
    print(dog, result)
binList = [0,1,1,0,1,1,0]

# Add your code below
for item in [binList]:
    if (item == 1):
        print("One")
    else:
        print()

    
for i in binList:
    if i == 1:
        print('One')
for item in [dogList]:
    if (item == "Bulldog" or "bulldog"):
        print("Bulldog")
    else:
        print()
        
word = "Bulldog"
for dog in dogList:
    if dog.lower().find(word.lower()) != -1:
        print(dog)

for dog in dogList:
    if word.lower() in dog.lower():
        print(dog)
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.


# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
    words = line.split()
    numWords += len(words)
    
    line = line.rstrip()
    numChars += len(line)

    numlines += 1
    
# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below