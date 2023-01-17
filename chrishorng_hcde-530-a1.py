lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']
s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)

print(lst[0])

#b.	Print the last element of otherlst

print(otherlst [-1])

#c.	Print the first five elements of lst

print(lst [0:5])

#d.	Print the fifth element of otherlst

print(otherlst[4])

#e.	Print the number of items in lst

print(len(lst))

#Exercise 2 (working with a string):
#a.	Print the first four characters of s

print (s[0:4])

#b.	Using indexing, print the substring "test" from s

print (s[10:14])

#c.	Print the contents of s starting from the 27th character (H)

print (s[26:])

#d.	Print the last three characters of s

print (s[-3:])

#e.	Print the number of characters in s

print (len(s))
# definied a variable to store the value 

fac=13

#defined a variable to store the value

#got stuck here!
#defined variable with string values
x = "Happy" 
y = "New" 
z = "Year!"

print (x,y,z)

# Defining functions 

def s1 ():
    x = "Happy"
    return x

def s2 ():
    y = "New"
    return y
    
def s3():
    z = "Year!"
    return z 


# Executing the three functions

print (s1 ())
print (s2 ())
print (s3 ())


# Defined one function to take 3 parameters 

def s0(x,y,z):
    print (x,y,z)
    
s0(s1(), s2 (), s3 ())

# Defined function to take two parameters

def add (x,y): 
    
    # Defined variable to store sum of 2 numbers 
    
    sum = x + y
    

    
    print (x, "+", y, "=", sum)
    
      
add (3,4)
    
