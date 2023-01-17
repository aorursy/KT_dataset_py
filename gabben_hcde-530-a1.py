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
len(lst)

#Exercise 2 (working with a string):
#a.	Print the first four characters of s
print(s[0:4])

#b.	Using indexing, print the substring "test" from s
print(s[10:14])

#c.	Print the contents of s starting from the 27th character (H)
print(s[26:34])

#d.	Print the last three characters of s
print(s[31:34])

#e.	Print the number of characters in s
len(s)

10+3
happy = "Happy"
new = "New"
year = "Year!"
print (happy, new, year)

def happy():
    return "Happy "
def new():
    return "New "
def year():
    return "Year!"
happy() + new() + year()
def hpn():
    return happy() + new() + year()
hpn()
def total(x, y):
    print(x, ' + ', y, ' = ', x + y)
    return (x + y)
total(3,4)
