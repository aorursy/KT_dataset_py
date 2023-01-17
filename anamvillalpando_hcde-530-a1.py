lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']

s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)
print(lst[0])
#b.	Print the last element of otherlst
print(otherlst[-1])
#c.	Print the first five elements of lst
print(lst[0:6])
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
# Calculates the factorial of the given number.
def fact(n):
    f = 1
    while(n>1):
        f=f*n
        n-=1
    return f

print ("13!=", fact(13))
h = 'Happy'
n = 'New'
y = 'Year!'

print(h,n,y) # OR print(h + ' ' + n + ' ' + y)
# Returns 'Happy'
def h():
    return 'Happy'
# Returns 'New'
def n():
    return 'New'
# Returns 'Year!'
def y():
    return 'Year!'

print(h())
print(n())
print(y())
# Returns 'Happy'
def h():
    return 'Happy'
# Returns 'New'
def n():
    return 'New'
# Returns 'Year!'
def y():
    return 'Year!'

print(h(),n(),y()) # OR print(h() + ' ' + n() + ' ' + y())
# Adds 2 numbers and prints its sum in equation form.
def add(a,b):
    sum = a + b
    print(a,"+",b,"=",sum)

# Adds 2 numbers and prints its sum in equation form.
def add2(a,b):
    sum = a.__add__(b)
    print(str(a) + " + " + str(b) + " = " + str(sum))
    
add(3,4)
# add2(3,4)