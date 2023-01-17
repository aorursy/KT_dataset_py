mylist = [1, 2, 3, 4, 5, 6]

for element in mylist:

    print(element)

    

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for xyz in mylist:

    total = total + xyz

print (total)

    

s = "This is a test string for HCDE 530"

# Add your code below

# Splits at space 

stringlist = s.split()

for element in stringlist:

    print(element)
# Add your code here

# Im not sure if this is a pythonic solution , but range and a foor loop can help in a bigger list

for i in range(len(mylist)):

    if mylist[i] == 4:

            mylist[i] = "four"

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

    print(line.strip())





# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound ","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]





# Add your code below

# this is my first attempt to solve this problem , although this works . if one of the elements in the list have two terrier one capital one small , it can cause a problem as the the if condition start with the capital one

 #for element in  dogList:

    #capitalTerrier = element.find('Terrier')

  #  smallTerrier = element.find('terrier')

   # if(capitalTerrier > -1):

    #    print(capitalTerrier)

  #  elif(smallTerrier > -1):

   #     print(smallTerrier)

 #   else

     #   print(-1)



#Another option, smaller but more confusing , this method worked here but it surely altered the element before using the find() which might not be ideal in other situations

for element in dogList:

    lower = element.lower()

    print(lower.find('terrier'))



binList = [0,1,1,0,1,1,0]



# Add your code below

for i in range(len(binList)):

    if binList[i] == 1:

            binList[i] = "One"

            print(binList[i])
# Add your code below

for Element in  dogList:

    lower = Element.lower()

    CBulldog = lower.find('bulldog')

    if(CBulldog > -1):

            print(Element)



 
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

#Iterating over a file as in for line in file breaks the file into lines. line.split() turns a line into words

for line in fname:

    words = line.split()

    numLines  += 1

    numWords += len(words)

    numChars += len(line)





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



fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



for line in fname:

    words = line.split()

    numLines  += 1

    numWords += len(words)

    numChars += len(line)

    

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



fname.close()