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
len(lst)

#Exercise 2 (working with a string):
#a.	Print the first four characters of s
print(s[0:4])

#b.	Using indexing, print the substring "test" from s
print(s[10:15])

#c.	Print the contents of s starting from the 27th character (H)
print(s[26:])

#d.	Print the last three characters of s
print(s[31:])

#e.	Print the number of characters in s
len(s)

print(13*12*11*10*9*8*7*6*5*4*3*1)
print("Happy", "New", "Year!")  
print('Happy New Year!')
print('Happy', 'New', 'Year!', sep=None)
h = "Happy"
n = "New"
y = "Year!"
print(*(h,n,y))
h = "Happy"
n = "New"
y = "Year!"

print(h,n,y)
def sum(x,y):
    print(x,"+",y,"=",x+y)
sum(3,4)