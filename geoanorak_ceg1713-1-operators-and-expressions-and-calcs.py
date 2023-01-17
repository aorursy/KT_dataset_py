# A Python comment starts with a # symbol
# in this session we will examine the use of simple python operators and expression
# An operator is a (mathematical) symbol that defines an operation between 2 bits of data eg "+" is the addition operator
# An expression is another word for a calculation or set of operators that manipulate data to create a new result = 5 + 4 is an expression
5 + 4
10 - 3.7
10 / 3
45.3 - 45
round(45.3-45, 4)

100 * 34
10 // 3
10 % 3
10 #integer - whole number
10.7 #Float (floating point number / real number)
"text"  #called Strings
'more text' #can use single quotes but can't mix and match
True  #Boolean - can only be True or False
False
10 + 17.5
10 + 'text'  #creates an error
10 * "hello"
10 + 3 * 5
(10 + 3) * 5
#round takes 2 arguments - the first is a float to round and 
#the 2nd is an integer that specifies how many decimal places to round to

round(45.6667251, 2)
round(45.67874, 1.5)
import math #only need 1 import statement per session (not everytime you use it!)

math.pow(3, 2) 
pow(3.14, 3)
pow(3.14, 1.5)
math.sqrt(9) #the square root
math.sin(90) #Should be 1
math.radians(90)
math.sin(math.radians(90))

math.atan(4/7)
math.degrees(math.atan(4/7))
import math
round(math.degrees(math.atan(12.586/15)), 4)


round(math.degrees(math.atan(15/12.586)), 4)


math.sqrt(math.pow(12.586, 2)+ math.pow(15, 2))