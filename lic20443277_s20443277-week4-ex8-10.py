#Exercise 8

In=list(map(float,input('Please enter the numbers splitted by commas",": ').split(',')))
print("The number list you entered is: ", In)
    
def Multiply(InputList):
    """A Python function to multiply all the numbers in a list."""
    Substrate=1
    
    while InputList:
        Substrate*=InputList[0]
        del InputList[0]   
    print("The result reached by multiplying each item in the list is: ",Substrate)

Out=Multiply(In)
## Exercise 9

Test=float(input("Please enter the number for testing: "))
Min=float(input("Please enter the minimum value of the test range: "))
Max=float(input("Please enter the maximum value of the test range: "))

while Min>Max:
    print("The range you set is invalid.")
    Min=float(input("Please re-enter the minimum value of the test range: "))
    Max=float(input("Please re-enter the maximum value of the test range: ")) 

def Range_Test(Test,Min,Max):
    """A Python function to check whether a number is in a given range"""
    if Test>=Min and Test<=Max:
        print(Test," is in the range from ",Min," to ",Max,".",sep="")
    else:
        print(Test," is NOT in the range from ",Min," to ",Max,".",sep="")
    
Range_Test(Test,Min,Max)
#Exercise 10

Text=input("Please enter the text for case-sensitive analysis: ")

def Letters(Text):
    """A Python function that accepts a string and calculate the number of upper case letters and lower case letters."""
    List=list(Text)
    Upper=0
    Lower=0
    for A in List:
        if A.isupper():
            Upper+=1
        if A.islower():
            Lower+=1
    print("The text contains",Upper,"upper case letter(s) and",Lower,"lower case letter(s).")
        
Letters(Text)
