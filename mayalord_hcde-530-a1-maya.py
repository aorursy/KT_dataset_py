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

print(s[26:34])



#d.	Print the last three characters of s

print(s[31:34])



#e.	Print the number of characters in s

print(len(s))

cal = 7  + 6

print(cal)



#This variable  is  storing an equation that equals 13. 

#That varialbe is then printed
h  = "Happy "

n = "New "

y = "Year!"

print(h + n + y)



#Each of the variable is storing part of the "Happy New  Year!" string.

#When the variable are added the print the entire string
def h():

    print("Happy ")



def n():

    print("New ")

    

def y():

    print("Year!")

    

h()

n()

y()



#Each of the functions print part  of  the "Happy New Year!" string. 

#When  they are each printed the entire string is returned
def hny():

    def h():

        print("Happy ")

    h()

    def n():

        print("New ")

    n()

    def y():

        print("Year!")

    y()

    

hny()



#The main function hny() is each of the previously written functions.

#When the main function is printed the entire string is returned



def q(v,x):

    a = v + x

    print(v, "+", x, "=", a)    

q(1,3)

    

#This uses two variable v and x. 

#Withing the function variable a is defined using v and x

#When the two outputs are given to the function, the variable a is able to be defined