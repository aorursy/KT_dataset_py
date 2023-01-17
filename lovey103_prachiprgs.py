#Write a program to find if a number is prime or not ,
#if the number is prime then print the table i.e number multiplied till 10.

#Input would be integer .

prime_num=int(input("Enter the number::"))

if prime_num>1:
    if (prime_num % 2) == 0 or (prime_num%3)==0 or (prime_num%5)==0 or (prime_num%7)==0:
        print("Number is NOT PRIME")
    else:
        print("Number is PRIME\n")
        print("Table of " ,prime_num,"is\n")
        for j in range(1,11):
            print(prime_num,"x",j," =",prime_num*j)
else: print("Number is not prime")
#Using dictionary take input of students name and their marks in english , 
#hindi maths find percent , rank based upon marks entered 
    
num=int(input("Enter number of students::"))

stud={}
i=0
for i in range(num):
    name=input("Enter name::")
    eng=int(input("Enter english marks out of 10::"))
    hindi=int(input("Enter hindi marks out of 10::"))
    maths=int(input("Enter maths marks out of 10::"))
    percent=((eng+hindi+maths)/30)*100
    stud.update({name:int(percent)})
print(stud)
max_key=max(stud, key=stud.get)
print("The hightest rank holder is::",max_key,max(stud.values()))
#Take input of string,Count No of Vowels ,Consonants and Special Character
#print the No of Vowel , Consonants and special characters( Blanks would be part of Special Character)
#And Winner which has max no'

text=[]
numvowel=[]
numcon=[]
numsp=[]

count=0
i=0
vowel=['a','e','i','o','u','A','E','O','I','U']
special=['!','@','~','#','$','%','^','&','*','(',')','|','<','>','/','{','?','}','[',']',' ']

text=str(input("Enter the string::"))
listlen=len(text)

for word in range(listlen):
    if text[word] in vowel:
        numvowel.append(word)   
    elif text[word] in special:
        numsp.append(word) 
    else:
        numcon.append(word)
            
vnum=len(numvowel)
cnum=len(numcon)
snum=len(numsp)
print("NO. OF VOWELS::",vnum)
print("NO. OF CONSONENTS::",cnum)
print("NO. OF SPECIAL CHARACTERS::",snum)
maxnum=max(vnum,cnum,snum)
if vnum==cnum==snum:
    print("All ARE WINNERS")
elif maxnum==vnum:
    print("VOWEL WON")
elif maxnum==cnum:
    print("CONSONENT WON")
else:
    print("SPECIAL WON")
#Using Python OS command.Take input of directory name with path,Check if the directory path exist 
#If not create the directory.Create a file in directory and write "This is my first file writing in python in it "

import os

dirName=input("Enter the path of the directory(home:/users/...)::")

if not os.path.exists(dirName):
    print("Directory " , dirName ,  "does not exist!!")
    yesno=input("Do u want to create a directory?(Enter Y or N)::")
    if yesno=="Y":
        os.mkdir(dirName)
        filename="myfile.txt"
        filepath=os.path.join(dirName,filename)
        f=open(filepath,'w')
        f.write("this is my first file")
        f.close()
        print("Directory " , dirName ,  "Created ")
        print("File ", filepath ,  "Created ")
else:    
    print("Directory " , dirName ,  " already exists")
    
#read data from the file

import json

with open('c:/temp/myfile.json') as f:
    data = json.load(f)
print(data)
#write data to the file

import json

person_dict = {"name": "prachi",
"languages": ["hindi", "urdu"],
"married": True,
"age": 41
}

with open('c:/temp/myfile1.json', 'w') as json_file:
    json.dump(person_dict, json_file)