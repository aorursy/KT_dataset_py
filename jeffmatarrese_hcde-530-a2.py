mylist = [1, 2, 3, 4, 5, 6]

for i in mylist: 

    print(i)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for i in mylist:

    total = total + i

    

print(total)
s = "This is a test string for HCDE 530"

# Add your code below

s_list = s.split()

for i in s_list:

    print(i)
# Add your code here

# making mylist2 in case I mess it up

mylist2 = mylist 

for i in mylist2:

    if i == 4: #check condition

        mylist2[i] = 'four' #if yes change value

mylist2



# It worked!
# The code below allows you to access your Kaggle data files

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/test.txt', 'r')

for line in fname:

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for dogs in dogList:

    index = dogs.find('terrier' or 'Terrier') #index local variable set to hold the .find() result

    print(index)
binList = [0,1,1,0,1,1,0]



# Add your code below

for bin in binList:

    if bin == 1:

        print("One")

for pupper in dogList:

    if pupper.find('Bulldog') != -1: #simplest way I could think of. Checks result of .find(), if it has any value besides -1 item is printed.

        print(pupper)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



for line in fname:

    numLines = numLines + 1 #iterating through text counts lines first, +1 to line number

    w = line.split() #split each line into word list

    numWords = numWords + len(w) #count word list length

    for word in w:

        numChars = numChars + len(word) #iterate through word list and count the length of each item

    



# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below



numChars = 0

numLines = 0

numWords = 0



f_sher = open('/kaggle/input/sherlock.txt', 'r')



for line in f_sher:

    numLines = numLines + 1

    w = line.split()

    numWords = numWords + len(w)

    for word in w:

        numChars = numChars + len(word)

    

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



f_sher.close()