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

print(s[0:4])

#b.	Using indexing, print the substring "test" from s

print(s[10:14])

#c.	Print the contents of s starting from the 27th character (H)

print(s[26:])

#d.	Print the last three characters of s

print(s[-3:])

#e.	Print the number of characters in s

print(len(s))
#Since we would be multiplying every number from 13 onwards till 1, we initialize the final result as 1 since multiplication by 0 is zero.

factorial = 1

#Starting from 13, and decreasing by 1 every time

for num in range(13, 0, -1):

    factorial *= num

print(factorial)
var1 = 'Happy'

var2 = 'New'

var3 = 'Year!'

print(var1 + " " + var2 + " " + var3)
def happyStr():

    return "Happy"

def newStr ():

    return "New"

def yearStr():

    return "Year!"



happyStr()

newStr()

yearStr()



#Note: I think the reason that we don't see the entire string of 'Happy New Year!' together is because we are just returning and calling the function, not printing it. Moreover, Python is a sequential language and it cannot execute 3 functions concurrently; hence, we only see the final function call that prints 'Year!'
def combine():

    finalStr = happyStr() +" " + newStr() + " " + yearStr()

    print(finalStr)

combine()



#In this case, the function 'combine()' prints the result, instead of returning the final string
def add(x, y):

    result = x + y

    return "%d + %d = %d" % (x, y, result)

add(12, 14)

    