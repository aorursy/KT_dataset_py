lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']
s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)
print(lst[0])
#b.	Print the last element of otherlst
print(otherlst[6])
#c.	Print the first five elements of lst
print(lst[0:5])
#d.	Print the fifth element of otherlst
print(otherlst[4])
#e.	Print the number of items in lst
print(len(lst))

#Exercise 2 (working with a string):
#a.	Print the first four characters of s
print(s[0:4])
#b.	Using indexing, print the substring "test" from s
print(s[10:14])
#c.	Print the contents of s starting from the 27th character (H)
print(s[26:])
#d.	Print the last three characters of s
print(s[-3:])
#e.	Print the number of characters in s
print(len(s))
def factorial(i):
    k = 1
    while ( i>=1 ):
        k = k * i
        i = i - 1
        print(k)

factorial(13)
a = "Happy"
b = "New"
c = "Year!"
print(a, b, c)
def first():
    print("Happy")
def second():
    print("New")
def third():
    print("Year!")

first()
second()
third()
def newf():
    first(), second(), third()

newf()
def add(a,b):
    print(a, "+", b, "=", a+b)

add(3,4)