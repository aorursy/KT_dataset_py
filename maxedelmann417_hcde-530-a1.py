lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 

otherlst = ['a','b','c','d','e','f','g']

s = "This is a test string for HCDE 530"



#Exercise 1 (working with a list):

#a.	Print the first element of lst (this one has been completed for you)

print(lst[0])

#b.	Print the last element of otherlst

print(otherlst[-1])

#c.	Print the first five elements of lst

print(lst[:5])

#d.	Print the fifth element of otherlst

print(otherlst[4])

#e.	Print the number of items in lst

print(len(lst))

#Exercise 2 (working with a string):

#a.	Print the first four characters of s

print(s[:4])

#b.	Using indexing, print the substring "test" from s

for xyz in "test":    print(xyz)

#c.	Print the contents of s starting from the 27th character (H)

print(s[27:])

#d.	Print the last three characters of s

print(s[-3:])

#e.	Print the number of characters in s

print(len(s))
#I used the print function and simply multiplied each number within the parathesis to output 13 factorial

print(13*12*11*10*9*8*7*6*5*4*3*2*1)

#I created each string variable to represent a different word and used the print function to add them together to create the phrase "Happy New Year!"

str1 = "Happy"

str2 = " New"

str3 = " Year!"

print(str1 + str2 + str3)
#Each function only outputs one word as a string in the phrase "Happy New Year", when executed at the same time, they output the entire phrase.

#I was unsure how to output them all on the same line. When I tried adding them together through the print function, I would get error messages.



def f1():

    print("Happy")



def f2():

    print("New")

    

def f3():

    print("Year!")

    

f1() 

f2()

f3()

#I created the "greeting" function by creating 3 string variables and using an equation to form them into a phase. 

#I first tried creating a separate function for each word, but I couldn't figure out how to add them all together on the same line. 

def greeting():    

    f1 = "Happy"

    f2 = " New"

    f3 = " Year!"

    x = f1 + f2 + f3

    return(x)



print(greeting())
#i1 and i2 represent the integers that would be plugged into the equation and x represents the sum of the total of i1 and i2. 

def add(i1, i2):    

    x = i1 + i2

    return(x)



print(add(3,4))
