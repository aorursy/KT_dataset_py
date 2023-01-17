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

for xyz in "test":

    print(xyz)

#c.	Print the contents of s starting from the 27th character (H)

print(s[27:])

#d.	Print the last three characters of s

print(s[-3:])

#e.	Print the number of characters in s

print(len(s))
#I used print to multiply 13 factorial, in which every number is decreasing by 1 and multiplied to each other.



print(13*12*11*10*9*8*7*6*5*4*3*2*1)
#In order to use 3 different strings, I seperate each of the words into a different string. To add a space in between each word, I added a blank space after "Happy" and "New". Then I used the print function to add the variables together. 



str1= "Happy "

str2= "New "

str3= "Year!"

print(str1+str2+str3)
#I created a function to define "Happy" through the letter h

def f1():

    h = "Happy"

    return h

    

#I created a function to define "New" through the letter n

def f2():

    n = "New"

    return n

    

#I created a function to define "Year!" through the letter y

def f3():

    y = "Year!"

    return y



#then I printed each function individually, to get "Happy New Year" on seperate lines.

print(f1())

print(f2())

print(f3())
#Using the functions from the previous exercise, I defined hny. So I printed the functions all together by using the previously defined function name. 

def hny():

    print(f1(), f2(), f3())

    

hny()
#Honestly not sure how do to this, I had to look up some instruction on how to do this. ALso not sure why I'm getting "none" in the output.

def add(x,y):

    print(x, "+", y, "=", x+y)

    

print(add(3,4))