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
print(s[0:4])
      
#b.	Using indexing, print the substring "test" from s
print(s[10:14])

#c.	Print the contents of s starting from the 27th character (H)
print(s[26:])

#d.	Print the last three characters of s
print(s[-3:])

#e.	Print the number of characters in s
print(len(s))
#if I were just using python as a calculator I could calculate the fatorial as follow:
factorial_13 = 1*2*3*4*5*6*7*8*9*10*11*12*13
print(factorial_13)

print("\n")




#based on some internet searching I found some help with code that would be more efficient for finding factorials using loops.
n=13
result = 1

for i in range(1,n+1):
    result = result*i
print("factorial of", n, "is", result)

 
mystring = "Happy" + " New" + " Year!"
print(mystring)

def greet_happy():
    return 'Happy '


def greet_new():
    return 'New '


def greet_year():
    return 'Year!'


print(greet_happy()+''+greet_new()+''+greet_year())

        









    
    

def greet_hny():
    return (greet_happy()+''+greet_new()+''+greet_year())

print(greet_hny())

def sum_of_numbers(a, b):
    print (a, '+', b, '=', a+b)
    
sum_of_numbers(3,4)



#2nd attempt - tried doing it using return in the function
x=6
y=8

def sum_of_num(x, y):
    return (x+y)

print(x, '+', y, '=', sum_of_num(x, y))



