print("Apples are better than \n Oranges")
ishan = "I love python because it is so easy to use it and also it is used by many other IT companies"
print(ishan[ishan.find("it"):])
#print(ishan[0:13])
str_ex = "I Love Pyton"
print(ishan)
print (ishan.replace("I love python",str_ex))


str_input = "Hey how are you doing. I am doing great"
print(str_input[0:10] +"_" +  str_input[-10:])
string1 = "Hotelspace"
string2="Facilities"
print(string1[0:3]+"@"+string1[-3:])
print(string2[0:3]+"@"+string2[-3:])
strr = "Hello World"
print(strr[-5:-1])

str_reverse="5+63-45"
print(str_reverse)
#print(str_reverse[-2:]) = 45
#print(str_reverse[-3:-2]) = -
#print(str_reverse[-4:-3])=3
#print(str_reverse[-5:-4])=6
#print(str_reverse[1:2])=+
#print(str_reverse[0:1])= 5
print(str_reverse[-2:]+str_reverse[-3:-2]+str_reverse[-4:-3]+str_reverse[-5:-4]+str_reverse[1:2]+str_reverse[0:1])

from difflib import SequenceMatcher as sq

str_s = "India is a great country with a lot of heritage"
str_x = "South Africa is a great country with a lot of freedom"
print(str_s.replace("is a great country with a lot of","") + " "+ str_x.replace("is a great country with a lot of",""))
#print(str_x)

variable1="This is a test to check the unique characters in the string"

def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count 


lst=list(variable1.lower())
my_list = list(set(lst))
#print(lst)
for item in my_list:
    print(countX(lst,item) , item)

var3 = "abcdefgh"
print(var3)
