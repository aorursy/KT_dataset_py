lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']
s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)
print(lst[0])
#b.	Print the last element of otherlst
print(otherlst[-1])
#c.	Print the first five elements of lst
print(lst[0:5])
#d.	Print the fifth element of otherlst
print(otherlst[4])
#e.	Print the number of items in lst
print(len(lst))
#Exercise 2 (working with a string):
#a.	Print the first four characters of s
print(s[:4])
#b.	Using indexing, print the substring "test" from s
print(s[10:14])
#c.	Print the contents of s starting from the 27th character (H)
print(s[26:])
#d.	Print the last three characters of s
print(s[-3:]) 
#comment: print(s[-3:0]) also works

#e.	Print the number of characters in s
print(len(s))
a=1
for factorial in range(1,14):
    a=a*factorial
print (a)

    
string1="Happy "
string2="New "
string3="Year!"
print(string1,string2,string3)
#print(string1+string2+string3) also works.
def a():
    print ("Happy")
def b():
    print ("New")
def c():
    print ("Year!")
a()
b()
c()


def a():
    return "Happy "
def b():
    return "New "
def c():
    return "Year!"
print (a() + b()+ c())
def happylastquestion(a, b):
    print(a, "+", b, "=",a+b)
happylastquestion(1,3)