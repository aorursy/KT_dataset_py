mylist = [1, 2, 3, 4, 5, 6]
for food in mylist:
    print("turnips are actually pretty tasty when cooked the right way")
 #pure sidebar commentary on this additional code beyond the task of Step 1-- I couldn't get rid of that space before "have" below   
print("\n","have you tried them cooked with just a little butter and sugar?")
total=0
for x in mylist:
    total=total+x
print(total)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0
#I could not reproduce this without looking up the code
s = "This is a test string for HCDE 530"
# Add your code below
#I first tried
#splits=s.split()
#for x in ssplits:
    #print(x)
#but then got the error, "name'ssplits' is not defined"
for x in s.split():
    print(x)

#I tried print(mylist.replace("4","four")) 
#and print(mylist.replace([3],"four"))
#and both of those versions with the print operator removed and then added in as an additional line
#for all of those, I keep getting the error, 'list' object has no attribute 'replace'
#realizing that this is a list and not a string, I revised the code

del(mylist[3])
mylist.insert(3,"four")
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
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below
for x in dogList:
    if x.endswith("terrier")== True: #I couldn't figure out a contain function, so I looked it up and found the endswith function. This code wasn't working, and I messed with it for 45 minutes until I found was just missing the colon at the end. ARRGHH!!
        y=x.find("terrier") #I also had to look up how to use the find function
        print(y)
    elif x.endswith("Terrier")== True: #I don't remember if python cares about capitalization, so I threw this in
        z=x.find("Terrier")
        print(z)
    else: #anything without "terrier" or "Terrier" do this
        print("-1")
binList = [0,1,1,0,1,1,0]

# Add your code below
for x in binList:
    if x==1:
        print("One")
#I initially tried this code below, but returned nothing
#for x in dogList:
    #if x.find("Bulldog")== True:
        #print(x)
#So, I replaced the command in the second line with x.endswith
for x in dogList:
    if x.endswith("Bulldog")== True:
        print(x)
   
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data
fname = open('/kaggle/input/a2data/test.txt', 'r')

# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line in fname:
    line=line.strip("\n") #return the string of characters in between newline markers
    words=line.split() #split the string line into substrings by the spaces that delineate them
    numLines=numLines+1 #accumulation pattern for number of lines
    numWords=numWords+len(words)#accumulation pattern for number of words
    numChars=numChars+len(line)#accumulation pattern for number of chars

# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')
print()

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
#It is late, and I am having a hard time understanding why this code would be different than the code for the prompt above, outside of updating the variable names and clarifying the definitions, so here goes:
numChars2 = 0 #chars definition includes alphanumeric characters, spaces, punctuation, and symbols
numLines2 = 0
numWords2 = 0 #as in the previous prompt, something that is delimited by any whitespace, so that the number of words on a line that is read from a file is simply the number of strings returned by split().

# create a file handle called fname2 to open and hold the contents of the data file
fname2 = open('/kaggle/input/a2data/sherlock.txt', 'r')


# Add your code below to read each line in the file, count the number of characters, lines, and words
# updating the numChars, numLines, and numWords variables.
for line2 in fname2:
    line2=line2.strip("\n") #return the string of characters in between newline markers
    words2=line2.split() #split the string line into substrings by the spaces that delineate them
    numLines2=numLines2+1 #accumulation pattern for number of lines
    numWords2=numWords2+len(words2)#accumulation pattern for number of words
    numChars2=numChars2+len(line2)#accumulation pattern for number of chars

print(f'characters {numChars2}')
print(f'lines {numLines2}')
print(f'words {numWords2}')
print()

fname2.close()