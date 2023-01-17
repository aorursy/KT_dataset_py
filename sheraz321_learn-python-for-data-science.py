def palindrome(str):

    index = 0
    l = len(str)

    while (index) < l / 2 + 1:
        if str[index] != str[(-index) - 1]:
            return False 
        index =index+ 1
    else:
        return True
    
s = "dad"
flag=palindrome(s);
if flag==False:
    print ("Not Palindrome")
else:
    print("palindrome")

s = "mad"
flag=palindrome(s);
if flag==False:
    print ("Not Palindrome")
else:
    print("palindrome")
    
def fabnocci(num):
    
    num1= 0
    num2= 1
    num3=0 
    index=0
    if num==1:
        print(num1)
    
    elif num > 1:
        if index==0:
            print(num1)
            index+=1
        if index==1:
            print(num2)
            index+=1
        while index<num :
            num3=num1+num2
            num1=num2
            num2=num3
            print(num3)
            index+= 1



number = 5
fabnocci(number)
def substring_index(str1,str2):
    return str1.find(str2)


str1= "hello world"
str2= "world"
print(substring_index(str1,str2)) 

def anagrams(str1,str2):
    if sorted(str1)==sorted(str2):
        print("anagrams")
    else:
        print("not anagrams")


str1="godd"
str2="dog"
anagrams(str1,str2)
str1="god"
str2="dog"
anagrams(str1,str2)

