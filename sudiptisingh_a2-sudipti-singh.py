mylist = [1, 2, 3, 4, 5, 6]



#add a new variable called allthenumbers

for allthenumbers in mylist:

    #print the contents of the string

    print(allthenumbers)
#declare a value for the variable called total

total = 0

#add a new variable called allthenumbers

for allthenumbers in mylist:

    #add based on previous total

    total = total + allthenumbers

#print the value of the total variable at the end

print(total)

s = "This is a test string for HCDE 530"

#split string into words, which is coincidentally what I'm calling this variable

words = s.split()

#add a variable to refer to each word in the split version

for eachword in words:

    #print each word in the split version

    print(eachword)

#set the value of the fourth item (3rd index item) to equal "four"

mylist[3] = "four"

#print the string to see the revision

print(mylist)

#enables accessing Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#create a file handle to hold contents of data

fname = open('/kaggle/input/a2data/test.txt', 'r')



#create a variable for the function above

for line in fname:

    #print text as it is in the file by removing all trailing/newline characters

    print(line.rstrip())



#close the file

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



#add a new variable called adogsname

for adogsname in dogList:

    #include the lower case version of terrier

    if "terrier" in adogsname:

        #print the indexed location

        print(adogsname.find("terrier"))

    #include the upper case version of Terrier

    elif "Terrier" in adogsname:

        #print the indexed location

        print(adogsname.find("Terrier"))

    #print "-1" everywhere that the rules above don't apply

    else:

        print("-1")

binList = [0,1,1,0,1,1,0]



#add a new variable called number

for number in binList:

    #if something in the string is equal to 1, then print "One"

    if number == 1:

        print("One")

    #do nothing if the rules above don't apply

    else:

        pass

#add a new variable called adogsname

for adogsname in dogList:

    #include the lower case version of terrier

    if "bulldog" in adogsname:

        #print the indexed item

        print(adogsname)

    #include the upper case version of Terrier

    elif "Bulldog" in adogsname:

        #print the indexed item

        print(adogsname)

    #do nothing if the rules above don't apply

    else:

        pass
#declare values of variables to be used

numChars = 0

numLines = 0

numWords = 0



#create a file handle to hold contents of data

fname = open('/kaggle/input/a2data/test.txt', 'r')



#added my code below to read each line in the file and update variables to count characters, lines, and words

for line in fname:

    #split each line into words

    words = line.split()

    #count the number of words

    numWords = numWords + len(words)

    #count the number of lines

    numLines = numLines + 1

    #count the number of characters

    numChars = numChars + len(line)



#output code below is provided for you; you should not edit this

print(f'characters {numChars}')

print(f'lines {numLines}')

print(f'words {numWords}')



#close the file

fname.close()
#declare values of variables to be used

charactersInSherlock = 0

linesInSherlock = 0

wordsInSherlock = 0



#create a file handle to hold contents of data

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



#added my code below to read each line in the file and update variables to count characters, lines, and words

for line in fname:

    #split each line into words

    words = line.split()

    #count the number of words (these will all have new variable names so as not to overlap with the previous calculations)

    wordsInSherlock = wordsInSherlock + len(words)

    #remove trailing/newline characters

    line = line.rstrip()

    #count the number of lines

    linesInSherlock = linesInSherlock + 1

    #count the number of characters

    charactersInSherlock = charactersInSherlock + len(line)



#output code below is provided for you; you should not edit this

print(f'characters {charactersInSherlock}')

print(f'lines {linesInSherlock}')

print(f'words {wordsInSherlock}')



#close the file

fname.close()