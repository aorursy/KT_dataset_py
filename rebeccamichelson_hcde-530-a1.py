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

print(otherlst[5])

#e.	Print the number of items in lst

len(lst)

#Exercise 2 (working with a string):

#a.	Print the first four characters of s

print(s[0:4])

#b.	Using indexing, print the substring "test" from s

s.find('test')

#c.	Print the contents of s starting from the 27th character (H)

print (s[27:-1])

#d.	Print the last three characters of s

print(s[-3])

#e.	Print the number of characters in s

len(s)
num = 13

factorial = 1

if num < 0:

   print("Sorry, factorial does not exist for negative numbers")

elif num == 0:

   print("The factorial of 0 is 1")

else:

   for i in range(1,num + 1):

       factorial = factorial*i

   print("The factorial of",num,"is",factorial)
H = "Happy "

N = "New "

Y = "Year!"



print(H + N + Y)

def H():

    print("Happy")

    

def N():

    print("New")

    

def Y():

    print("Year!")

H()

N()

Y()
H + N + Y



def HNY():

    return H(), N(), Y()

#Call the function HNY

HNY()
def add(x,y):

    print(x+y)



#run function with inputs 3 and 4.

add(3,4)