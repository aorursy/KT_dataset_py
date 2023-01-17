#1: Write a program which will find all such numbers which are divisible by 7 
#   but are not a multiple of 5, between 2000 and 3200 (both included).


#Hints: Consider use range(#begin, #end) method

l1=range(2000,3201)

for x1 in l1:
    if (x1%7==0) and (x1%5!=0):
        print (x1)
#2. Write a program which can compute the factorial of a given numbers. 
#   The results should be printed in a comma-separated sequence on a single line. 
# input() function can be used for getting user(console) input
# raw_input() for python 2.x

#Suppose the input is supplied to the program:  8  
#Then, the output should be:  40320 
#Hints: In case of input data being supplied to the question, it should be assumed to be a console input. 

while True:
    try:
        num = int(input("Enter a number: "))
        break
    except:
        print("Please Enter a valid number")

fact = 1
result = []

if num < 0:
    print("Sorry, factorial does not exist for negative numbers")
elif num == 0:
    print("The factorial of 0 is 1")
else:
    for i in range(1,num + 1):
        fact = fact*i
        result.append(fact)
print("Factorial of ",num,"is ",result)
#3. With a given integral number n, write a program to generate a dictionary that contains (i, i*i) such that is an integral number between 1 and n (both included). and then the program should print the dictionary.
#Suppose the following input is supplied to the program: 8
#Then, the output should be: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}
#Hints: In case of input data being supplied to the question, it should be assumed to be a console input. Consider use dict()


while True:
    try:
        user_num = int(input("Enter a number: "))
        break
    except:
        print("Please Enter a valid number")
        
dict1 = {}
for io in range(1,user_num+1):
    dict2 = {io:io*io}
    dict1.update(dict2)
print(dict1)
#4. Write a program which accepts a sequence of comma-separated numbers from console and generate a list and a tuple which contains every number.
#Suppose the following input is supplied to the program: 34,67,55,33,12,98
    #Then, the output should be: ['34', '67', '55', '33', '12', '98'] ('34', '67', '55', '33', '12', '98')

#Hints: In case of input data being supplied to the question, it should be assumed to be a console input. tuple() method can convert list to tuple

user_input1 = int(input('Enter number of elements to be in list: '))
list1 = list(map(str,input('Enter elements inside list: ').split(',')))
tuple1 = tuple(list1)
print(list1,tuple1)
#5. Define a class which has at least two methods: getString: to get a string from console input and 
#printString: to print the string in upper case. Also please include simple test function to test the class methods.

#Hints: Use __init__ method to construct some parameters

class MyClass():
    def __init__(self):
        self.str1 = ""
    def getString(self):
        self.str1 = input("Enter String: ")
    def printString(self):
        print(self.str1.upper())

str1 = MyClass()
str1.getString()
str1.printString()
#6. Write a program that accepts a comma separated sequence of words as input and 
# prints the words in a comma-separated sequence after sorting them alphabetically.

# Suppose the following input is supplied to the program: without,hello,bag,world
# Then, the output should be: bag,hello,without,world

#Hints: In case of input data being supplied to the question, it should be assumed to be a console input.

y=input("Enter comma separated sequence of words : ")
y=sorted(y.split(","))
print(",".join(y))
#7. Write a program that accepts a sequence of whitespace separated words 
# as input and prints the words after removing all duplicate words and sorting them alphanumerically.
# Suppose the following input is supplied to the program: hello world and practice makes perfect and hello world again
# Then, the output should be: again and hello makes perfect practice world

#Hints: In case of input data being supplied to the question, it should be assumed to be a console input.
#We use set container to remove duplicated data automatically and then use sorted() to sort the data.

q1=input("Enter sequence of words : ")
q2=sorted(set(q1.split(" ")))
print(" ".join(q2))
#8. Write a program that accepts a sentence and calculate the number of upper case 
# letters and lower case letters.
#Suppose the following input is supplied to the program: Hello world!
#Then, the output should be: UPPER CASE 1 LOWER CASE 9

#Hints: In case of input data being supplied to the question, it should be assumed to be a console input.

word = input('Enter a word: ')
lower_count = sum(map(str.islower, word))
upper_count = sum(map(str.isupper, word))
print("UpperCase: ",upper_count," LowerCase: ",lower_count)
#9. A website requires the users to input username and password to register. Write a program to check the validity of password
#input by users. Following are the criteria for checking the password:
#1. At least 1 letter between [a-z]
#2. At least 1 number between [0-9]
#1. At least 1 letter between [A-Z]
#3. At least 1 character from [$#@]
#4. Minimum length of transaction password: 6
#5. Maximum length of transaction password: 12
#Your program should accept a sequence of comma separated passwords and will check them according to the above criteria. 
#Passwords that match the criteria are to be printed, each separated by a comma.
#Example. If the following passwords are given as input to the program: ABd1234@1,a F1#,2w3E*,2We3345
#Then, the output of the program should be: ABd1234@1

# You can use module called re
#Hints: In case of input data being supplied to the question, it should be assumed to be a console input.


import re
password=input("Enter comma separated sequence of words: ")
pass1=password.split(",")
list(pass1)
res=[]
pass_seq= "^.*(?=.{6,12})(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$]).*$"
for i in pass1:
    result = re.findall(pass_seq, i)
    if (result):
        res.append(result)
if not res:
    print("No Valid password found")
else:
    print(res)
#10. Python has many built-in functions, and if you do not know how to use it, you can read document online or find some books.
#But Python has a built-in document function for every built-in functions.
#Please write a program to print some Python built-in functions documents, such as abs(), int(), raw_input()
#And add document for your own function
    
#Hints: The built-in document method is __doc__

def SumNum(*args):
    '''
    This is document related to SumNum function.
    
    Function can be used to add multiple numbers.'''
    res=0
    for i in args:
        res += i
    return i
print(SumNum.__doc__)
print("\n"+"-"*100+"\n")
print(abs.__doc__)
print("\n"+"-"*100+"\n")
print(int.__doc__)
print("\n"+"-"*100+"\n")
print(input.__doc__)