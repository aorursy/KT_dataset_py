print("Hi there")
print("Let's have a picnic")
print("The lake is beautiful")

jorgeTheGoose("honk")
# Here is a comment. The interpreter sees that # symbol and says "not my problem, moving on!"

print("Let's visit campus") # This is an inline comment, which can be useful for making your code easy to understand 
print("\n\n\nThe lake is beautiful") # I put this in a second print statement, but I could have combined it with the first. \n is the code for a line break, by the way.

# jorgeTheGoose("honk")
# Integers
a = 1 # I can declare a variable by using =
b = -59
c = 12
d = a + b + c
print("d = "+ str(d)) # This is called 'string concatenation.' You can put strings together using the + operation. The str function converts objects into strings.
print("Its type is "+ str(type(d)))  # The type function returns the type of an object
# Floating point numbers (floats)
a = 1.                             # Add a decimal after an integer to make it a float
b = -59.23923                      # Floats can have high precision
c = 1.2e3                          # Floats understand scientific notation. This is equivalent to 1.2x10^3, aka 1200
d = a + b + c
print("d = "+str(d)) # Using str again to turn d into a string so that it can be combined with the string "d = "
print("Its type is "+str(type(d)))
# Int and int
2**6          # 2 to the 6th power
# Float and float
2.3 * 4.5
# Float and integer
4 * 9.8
# Division with ints
6/7
div = 20//7
remainder = 20%7
print("Integer division of 20//7 gives:")
print(div)
print("20 mod 7 gives the remander:")
print(remainder)
# Exercise 1 here

a = 4**3
b = 2 + 3.4**2
c = (1+3**6)**2
print(a)
print(b)
print(c)
string1 = "This is a string"            # double quotes
string2 = 'This is another string'      # single quotes
string3 = "This isn't a problem"        # single quote mark within double quotes
string4 = 'Here\'s another way to do it'# Escaped single within singles
# string5 = 'This won't work.'
print(string1)
print(string2)
a = "Howdy!"
print(a[3])
print(a[-3])
print(len(a))
b = "I like Mary Lyon"
b.upper() # returns an uppercase version
b.index("Mary") #returns the index of a string inside another
b.split() # returns a list of smaller strings. 
          # By default it splits on spaces, but you can also give it an input like b.split("y")
s = "Howdy!"
print(s[1])   # Takes one character starting at 1
print(s[4])   # Takes one character starting at 4
print(s[1:4]) # Takes all the characters between 1 and 4
print(s[3:])  # Takes all characters starting at 3 and going to the end
print(s[:3])  # Takes all characters until reaching 3
print(s[::2]) # Takes every other character
# Excercise 2

ex3 = "I do not get this!"

#### YOUR CODE HERE ####
notposition = ex3.index("not")
ex3[:notposition] + ex3[notposition+4:]
mylist=[1,2,3,4]
mylist[1:3] # Can address and slice just like a string
print("Pop() on a list of 1,2,3,4")
mylist = [1,2,3,4]
print(mylist.pop()) # list.pop() returns the last item in a list, and removes it from the list
print(mylist)

print("\nPop(0) on a list of 1,2,3,4")
mylist = [1,2,3,4]
print(mylist.pop(0)) # list.pop([index]) returns the indexed item in a list, and removes it from the list
print(mylist)
print("I can add to the end by concatenating a list at the end")
mylist = [1,2,3,4]
print(mylist+[5,6,7])

print("I can add to the beginning by concatenating a list at the start")
mylist = [1,2,3,4]
print([-2,-1,0]+mylist)
True and False
False and False
True or False
1 != 2
(1 != 2) and (5 > 5)
x = 3.7

# your solution here