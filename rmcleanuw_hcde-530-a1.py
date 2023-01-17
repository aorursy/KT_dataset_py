lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 

otherlst = ['a','b','c','d','e','f','g']

s = "This is a test string for HCDE 530"



#Exercise 1 (working with a list):

#a.	Print the first element of lst (this one has been completed for you)

print(lst[0])

#b.	Print the last element of otherlst

#I will use the negative command to access the last element.

print(otherlst[-1])



#c.	Print the first five elements of lst

#I will use the slice functionality we saw in class. The 0 starts with the first element of the list and the 5 tells it that I want to go up to, and including, the 5th element of the list.

print(lst[0:5])



#d.	Print the fifth element of otherlst

#When counting forward we start with 0, so the 5th element will be accessed with the int 4.

print(otherlst[4])



#e.	Print the number of items in lst

#I can use the len() command to count up the number of items in the list.

print(len(lst))



#Exercise 2 (working with a string):

#a.	Print the first four characters of s

#This will work the same as exercise 1c, but I will call the variable s and go to the 4th character rather than the 5th.

print(s[0:4])



#b.	Using indexing, print the substring "test" from s

#I'll setup what I'm looking for. I'll use the .index() fucntion to figure out where test is. Then I'll use the len() function to figure out how much beyond the index value I need to slice.

want = "test"

start = s.index(want)

end = start+len(want)

print(s[s.index(want):end])



#c.	Print the contents of s starting from the 27th character (H)

#27th character will be at index 27-1. I can use the slicing functionality and omit everything after the first colon to print through to the end of the list.

print(s[27-1:])



#d.	Print the last three characters of s

#I first tried using negative numbers, but realized you don't want the third from the last character and you don't want them in reverse order. So I will just grab the length and lob 3 off to start with the third from last. Then I can use the colon and follow with nothing to run through the end of the list.

print(s[len(s)-3:])



#e.	Print the number of characters in s

#Simple

print(len(s))
#Well I end up googling this wondering if there was a built in function. There is but you have to do this import math thing. I am wondering if you wanted us to actually define a little program similar to the counter program you showed in class, and I went back and re-watched that, but it seems like a very steep learning curve and I learned how to import and use this math funciton so I am calling that good learning for now :-) 

import math

math.factorial(13)
#I'm going to liberally append this to mean at least three different string variables and add a fourth for the space.

h="Happy"

n="New"

y="Year"

s=" "

h+s+n+s+y+"!"
h="Happy"

n="New"

y="Year"



def h1():

    print(h)



def h2():

    print(n)

    

def h3():

    print(y)

    

h1()

h2()

h3()
h="Happy"

n="New"

y="Year!"



#I am going to use the join function as part of my function. It will let me define a separator to place between the strings, which I will define as space.



def birthday():

    x = " ".join([h,n,y])

    print(x)

    



birthday()
def sumit(x,y):

    z = x+y

    numbers = [x,y,z]

    strings = str(numbers)

    out = " ".join([str(x),"+",str(y),"=",str(z)])

    print(out)



sumit(3,4)

sumit(1,2)

sumit(98,102)