lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']
s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)

#print the first one
print(lst[0])

#b.	Print the last element of otherlst

#print the one counted backward
print(otherlst[-1])

#c.	Print the first five elements of lst

#print 0-5
print(lst[:5]);
    
#d.	Print the fifth element of otherlst

#print 5th one
print(otherlst[4])

#e.	Print the number of items in lst

#count the length
print(len(lst))

#Exercise 2 (working with a string):
#a.	Print the first four characters of s

#print 0-4
print(s[:4])

#b.	Using indexing, print the substring "test" from s

#print 10-15
print(s[10:15])

#c.	Print the contents of s starting from the 27th character (H)

#print 27-last one
print(s[27:])

#d.	Print the last three characters of s

#print the last three counted backward
print(s[-3:])

#e.	Print the number of characters in s

#count the length
print(len(s))
#start from 1
factorial = 1

#use for loop to multiply all numbers
for i in range(1, 14):
    factorial = factorial*i
    print(factorial)
#print using separate strings
print("Happy" + " " + "New" + " " + "Year" + "!")

#Or we can
print("Happy", "New", "Year!")

#Or we can define them separately
a = "Happy"
b = "New"
c = "Year!"
print (a,b,c)
#define separate functions to return words
def H():
    p1 = "Happy"
    return p1
    
def N():
    p2 = "New"
    return p2
    
def Y():
    p3 = "Year!"
    return p3
    
#call the functions
print(H())
print(N())
print(Y())
#use a new function to call previous functions
def HNY(h,n,y):
    print(h,n,y)
    
HNY(H(),N(),Y())
#create a culculate function
def Cal(x, y):
    add = x + y 
    
    #print the statement while call the function
    print(x, "+", y, "=", add)
    
#call the function
Cal(5,6)