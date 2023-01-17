#
def funReturntype(param1):
    if param1.isnumeric()==False and param1.isalpha()==True:
        strOutput = "String"
    elif param1.isnumeric()==True:
        strOutput = "Integer"    
    elif param1.isalnum()==True :             
        strOutput = "Mixed String"
    elif param1.isalnum()==False:
        if re.findall("\w",param1):
            strOutput = "Mixed String"
        else:
            strOutput = "Special Characters"
            
    return strOutput
val = eval(input("Enter Value :"))
print(funReturntype(str(val)))
import re
import math
val = input("Enter No :")
if str(val).isnumeric() == True:
    intval =math.sqrt(int(val)) + 1
    cnt=0    
    for div in range(2, int(intval)):       
        rem = 0
        out = 0             
        rem = int(math.remainder(int(val), int(div)))
        out = int(int(val)/ int(div))
        
        if rem == 0 and out > 0 :
            cnt = cnt + 1 
            break
    if cnt == 0 and int(val) > 1: 
        print("Prime No :" ,val)
        for i in range(1,11):
            tblval = 1
            tblval = int(val) * int(i)
            print(tblval)
    else:
        print("This is not a Prime Number :", val)
else:
    print("Enter No")
noofS = input("enter no of Student ")
studict= {}
for i in range(int(noofS)):
    sname = input("Student Name:")
    m_english = input("English Marks:")
    m_hindi = input("Hindi Marks:")
    m_maths = input("Maths Marks:")
    total=int( m_english) + int(m_hindi) + int(m_maths)
    studict[i] = dict(sname=sname, m_english=m_english, m_hindi=sname,m_maths=m_maths,total=total)

    ##sorted()
print("----------------")
print("Name  " , "Percentage " , " Rank ")
print("----------------")
##for x in studict.items():  
  
  ##print(x[1]['sname'] ," " , x[1]['total'] ,"  " )
cnt=0
for x in sorted(studict.items(), key = 
             lambda kv:(kv[1]['total']),reverse=True):
    cnt=cnt +1
    print(x[1]['sname'] ," " , x[1]['total'] ,"  " , cnt )
val = input("Enter string :")
cntvow = 0

allEle = re.findall("[a-zA-Z]",val)
Vow = re.findall("[aeiou]",val)
Spcl = re.findall("[\W]",val)

conso = len(allEle) - len(Vow)
#print(allEle)
print("Consonants :" ,conso)
print("Vowel :", len(Vow))
print("Special :", len(Spcl))
if conso > len(Vow) and conso > len(Spcl):
    print("Winner Consonants")
if len(Vow) > conso and len(Vow) > len(Spcl):
    print("Winner Vowel")
if len(Spcl) > conso and len(Spcl) > len(Vow):
    print("Winner Special")  
import os
dirname = input("Enter Directory :")
if os.path.isdir(dirname) == True:    
    print(dirname)    
else:
    os.mkdir(dirname)
f = open(dirname + "/firstfile.txt", "w")
f.write("This is my first file writing in python in it")
f.close()
os.getcwd()

f = open("firstfile.txt", "r")
print(f.readline())
import json 
dt= {"Id":"1", "name":"Anika"}
dirname = input("Enter Directory :")
if os.path.isdir(dirname) == True:    
    print(dirname)    
else:
    os.mkdir(dirname)
f = open(dirname + "/jsonfile.json", "w")
json.dump(dt,f)
f.close()
f = open("jsonfile.json", "r")
print(f.readline())