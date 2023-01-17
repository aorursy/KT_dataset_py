mylist = [1, 2, 3, 4, 5, 6]
for number in mylist:
    print(number)
    
    

# Your notebook already knows about mylist. Sum its values by adding the code below this comment.

total = 0
for x in mylist:
    total = total + x
print(total)
    

s = "This is a test string for HCDE 530"
# Add your code below

list = s.split()
for words in list:
    print(words)


mylist[3] = "four"
print(mylist)


# The code below allows you to access your Kaggle data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# create a file handle called fname to open and hold the contents of the data file
fname = open('/kaggle/input/a3data/test.txt', 'r')

# Add your code below
# I had accidentally originally saved the data as a3data so I reflected that above. Even though I changed the data set name to A2 in Kaggle it appears that it is still recognizing the files as in an a3data file.
for line in fname:
    line = line.rstrip()
    print (line)
    
# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]

# Add your code below

for name in dogList:
    if 'Terrier' in name:
        print(name.find('Terrier'))
    elif 'terrier' in name:
        print(name.find('terrier'))
    else:
        print(-1)
        


   
     
   
    

binList = [0,1,1,0,1,1,0]

for x in binList:
    if x==1: 
        print('One')

    else:
        print("")
    
# Add your code below

for name in dogList:
    if 'Bulldog' in name:
        print(name)
    else:
        print('')
        

        

        
       
numChars = 0
numLines = 0
numWords = 0

# create a file handle called fname to open and hold the contents of the data file
# make sure to upload your test.txt file first, in a new dataset called a2data

# Add your code below to read each line in the file, count the number of characters, lines, and words
#I have tried to rename the dataset I uploaded to a2data but it is still being found at A3data.
fname = open('/kaggle/input/a3data/test.txt', 'r')



# updating the numChars, numLines, and numWords variables.
# Add your code below


for line in fname:
    wordlist = line.split()
    numLines = numLines + 1
    numWords = numWords + len(wordlist)
    numChars = numChars + len(line)
    


# output code below is provided for you; you should not edit this
print(f'characters {numChars}')
print(f'lines {numLines}')
print(f'words {numWords}')

# It's good practice to close your file when you are finished. This is in the next line.
fname.close()
# Add your code below

f = open('/kaggle/input/a3data/sherlock.txt', 'r')

numChar = 0
numLine = 0
numWord = 0


for lines in f:
    f_wordlist = lines.split()
    numLine = numLine + 1
    numWord = numWord + len(f_wordlist)
    numChar = numChar + len(lines)

f.close()

print(f'characters {numChar}')
print(f'lines {numLine}')
print(f'words {numWord}')


