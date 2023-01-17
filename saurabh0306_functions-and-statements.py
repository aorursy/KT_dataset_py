# A simple Python function to check 

# whether x is even or odd 

def evenOdd(x): 

	if (x % 2 == 0): 

		print("even")

	else: 

		print ("odd")



# Driver code 

evenOdd(2) 

evenOdd(3) 

def my_function(x):

  return 5 * x



print(my_function(3))

print(my_function(5))

print(my_function(9))
def tri_recursion(k):

  if(k>0):

    result = k+tri_recursion(k-1)

    print(result)

  else:

    result = 0

  return result



print("\n\nRecursion Example Results")

tri_recursion(6)
import math

# returning the exp of 4 

print ("The e**4 value is : ", end="") 

print (math.exp(4)) 



# returning the log of 2,3 

print ("The value of log 2 with base 3 is : ", end="") 

print (math.log(2,3)) 

import math

a = -10

b= 5



# returning the absolute value. 

print ("The absolute value of -10 is : ", end="") 

print (math.fabs(a)) 



# returning the factorial of 5 

print ("The factorial of 5 is : ", end="") 

print (math.factorial(b)) 

import statistics 



# initializing list 

li = [1, 2, 2, 3, 3, 3] 



# using median_grouped() to calculate 50th percentile 

print ("The 50th percentile of data is : ",end="") 

print (statistics.median_grouped(li)) 



# using mean() to calculate average of list elements 

print ("The average of list values is : ",end="") 

print (statistics.mean(li)) 



# using mode() to print maximum occurring of list elements 

print ("The maximum occurring element is  : ",end="") 

print (statistics.mode(li)) 

# python program to illustrate If statement 

  

i = 10

if (i > 15): 

   print ("10 is less than 15") 

print ("I am Not in if")
# python program to illustrate If else statement 

#!/usr/bin/python 

  

i = 20; 

if (i < 15): 

    print ("i is smaller than 15") 

    print ("i'm in if Block") 

else: 

    print ("i is greater than 15") 

    print ("i'm in else Block") 

print ("i'm not in if and not in else Block") 
# Python program to illustrate if-elif-else ladder 

#!/usr/bin/python 

   

i = 20

if (i == 10): 

    print ("i is 10") 

elif (i == 15): 

    print ("i is 15") 

elif (i == 20): 

    print ("i is 20") 

else: 

    print ("i is not present") 
# Python program to illustrate 

# while loop 

count = 0

while (count < 3):	 

	count = count + 1

	print("Hello Geek") 

# Python program to illustrate 

# Iterating over a list 

print("List Iteration") 

l = ["geeks", "for", "geeks"] 

for i in l: 

	print(i) 

	

# Iterating over a tuple (immutable) 

print("\nTuple Iteration") 

t = ("geeks", "for", "geeks") 

for i in t: 

	print(i) 

	

# Iterating over a String 

print("\nString Iteration")	 

s = "Geeks"

for i in s : 

	print(i) 

	

# Iterating over dictionary 

print("\nDictionary Iteration") 

d = dict() 

d['xyz'] = 123

d['abc'] = 345

for i in d : 

	print("%s %d" %(i, d[i])) 

# Function to convert number into string 

# Switcher is dictionary data type here 

def numbers_to_strings(argument): 

    switcher = { 

        0: "zero", 

        1: "one", 

        2: "two", 

    } 

  

    # get() method of dictionary data type returns  

    # value of passed argument if it is present  

    # in dictionary otherwise second argument will 

    # be assigned as default value of passed argument 

    return switcher.get(argument, "nothing") 

  

# Driver program 

if __name__ == "__main__": 

    argument=0

    print(numbers_to_strings(argument) )