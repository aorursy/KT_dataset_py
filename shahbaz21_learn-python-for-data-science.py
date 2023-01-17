def palindrome(str1):
    flag=True
    length=len(str1)
    size=length
    for i in range(int(length/2)):
        if str1[i]==str1[size-1]:
            size=size-1
            flag=True
            
        else:
            flag=False
            return flag
    return flag
        
    

str1='solos'
check=True
check=palindrome(str1)
if check==True:
    print('Palindrome')
else:
    print('Not palindrome')

def Anagram(str1,str2):
    length1=len(str1)
    length2=len(str2)
    str3=sorted(str1)
    str4=sorted(str2)
    flag=True
    
    if length1!=length2:
        flag=False
        return flag
    else:
        for i in range(length1):
            if str3[i]==str4[i]:
                flag=True
            else:
                flag=False
                return flag
            
    return flag

str1='listen'
str2='silent'
check=True
check=Anagram(str1,str2)
if check==True:
    print('Anagram')
else:
    print('Not Anagram')

def fibonaci(num):
    num1 = 0
    num2 = 1
    if num == 1:
        print("Fibonacci sequence upto",num,":")
        print(num1)
    else:
        print("Fibonacci sequence upto",num,":")
        for i in range(num):
            print(num1,end=' , ')
            add = num1+num2
            num1=num2
            num2=add

num = 6

fibonaci(num)

string = 'Python programming is fun.'


result = string.index('is fun')

print("Substring 'is fun':", result)