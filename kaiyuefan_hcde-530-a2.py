mylist = [1, 2, 3, 4, 5, 6]

for x in mylist:

    print(x)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0

for x in mylist:

    total=total+x

print(total)
s = "This is a test string for HCDE 530"

# Add your code below



for word in s.split():

    print(word)

#word=s.split()

#for x in word:

    #print(x)
# Add your code here

# i is index, n is valaue

for i,n in enumerate(mylist):

    if n==4:

        mylist[i]='four'

print(mylist)

# The code below allows you to access your Kaggle data files

# import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

# fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below



f=open('/kaggle/input/a2datanew/test.txt', 'r')

for line in f:

    line=line.strip()

    print(line)

f.close()



# It's good practice to close your file when you are finished. This is in the next line.

#fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



for words in dogList:

    

    if 'terrier' in words:

        print(words.find('t'))

    elif 'Terrier' in words:

        print(words.find('T'))

    else:

        print('-1')
binList = [0,1,1,0,1,1,0]



# Add your code below

for x in binList:

    if x==1:

        print('One')

    
for x in dogList:

    if 'Bulldog' in x:

        print(x)

    elif 'bulldog' in x:

        print(x)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

# fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.



f=open('/kaggle/input/a2datanew/test.txt', 'r')

for line in f:

    line=line.strip()

    numLines+=1

    # print(line)

    words=line.split()

    # print(words)

    numWords+=len(words)

    for x in words:      

        numChars+=len(x)

        





# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)

f.close()

# It's good practice to close your file when you are finished. This is in the next line.

# fname.close()
# Add your code below

numc=0

numw=0

numl=0

f=open('/kaggle/input/a2datanew/sherlock.txt', 'r')

for line in f:

    line=line.strip()

    numl+=1

    words=line.split()

    for words in line:

        numw+=len(words)

    for x in words:

        numc+=len(x)

print('%d characters'%numc)

print('%d lines'%numl)

print('%d words'%numw)

f.close()