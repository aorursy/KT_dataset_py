import numpy as np

import pandas as pd 



import os

print(os.listdir("../input"))
# Numeric Functions : 1) ceil()

#                     2) floor()

#                     3) fabs()

#                     4) factorial()

#                     5) copysign() 

#                     6) gcd()

#ceil() :- This function returns the smallest integral value greater than the number.

#          If number is already integer, same number is returned.

import math



a=5.8

b=5.2

c=5

print("{}\n{}\n{}\n".format(math.ceil(a),math.ceil(b),math.ceil(c)))
# floor() :- This function returns the greatest integral value smaller than the number. 

#            If number is already integer, same number is returned.

a=5.8

b=5.2

c=5

print("{}\n{}\n{}\n".format(math.floor(a),math.floor(b),math.floor(c)))
# fabs() :- This function returns the absolute value of the number.

a=-5.8

b=5.2

c=5

print("{}\n{}\n{}\n".format(math.fabs(a),math.fabs(b),math.fabs(c)))

# factorial() :- This function returns the factorial of the number. 

#                An error message is displayed if number is not integral.

a=8

b=2

c=5

print("{}\n{}\n{}\n".format(math.factorial(a),math.factorial(b),math.factorial(c)))
# copysign(a, b) :- This function returns the number with the value of ‘a’ but with the sign of ‘b’. 

#                   The returned value is float type.

a=9

b=-2

print(math.copysign(a,b))
# gcd() :- This function is used to compute the greatest common divisor of 2 numbers mentioned in its arguments. 

#          This function works in python 3.5 and above.

a=25

b=20

print(math.gcd(a,b))
# Operator Functions: 1) add()

#                     2) sub()

#                     3) mul()

#                     4) truediv()

#                     5) floordiv()

#                     6) pow()

#                     7) mod()

#                     8) lt()

#                     9) le()

#                     10) eq()

#                     11) gt()

#                     12) ge()

#                     13) ne()

# add(a, b) :- This functions returns addition of the given arguments. Operation – a + b.

import operator

a=4

b=8

operator.add(a,b)
# sub(a, b) :- This functions returns difference of the given arguments.Operation – a – b.

a=4

b=8

operator.sub(a,b)
# mul(a, b) :- This functions returns product of the given arguments.Operation – a * b.

a=4

b=8

operator.mul(a,b)
# truediv(a,b) :- This functions returns division of the given arguments.Operation – a / b.

a=4

b=8

operator.truediv(a,b)
# floordiv(a,b) :- This functions also returns division of the given arguments. 

# But the value is floored value i.e. returns greatest small integer.Operation – a // b.

a=9

b=4

operator.floordiv(a,b)
# pow(a,b) :- This functions returns exponentiation of the given arguments. Operation – a ** b.\

a=9

b=4

operator.pow(a,b)
# mod(a,b) :- This functions returns modulus of the given arguments. Operation – a % b.

a=5

b=6

operator.mod(a,b)
# lt(a, b) :- This function is used to check if a is less than b or not. 

# Returns true if a is less than b, else returns false. Operation – a < b.

a=5

b=6

operator.lt(a,b)
# le(a, b) :- This function is used to check if a is less than or equal to b or not. 

# Returns true if a is less than or equal to b, else returns false.Operation – a <= b.

a=5

b=4

operator.le(a,b)
# eq(a, b) :- This function is used to check if a is equal to b or not. 

# Returns true if a is equal to b, else returns false. Operation – a == b.

a=4

b=4

operator.eq(a,b)
# gt(a,b) :- This function is used to check if a is greater than b or not. 

# Returns true if a is greater than b,else returns false. Operation – a > b.

a=8

b=9

operator.gt(a,b)
# ge(a,b) :- This function is used to check if a is greater than or equal to b or not. 

# Returns true if a is greater than or equal to b, else returns false. Operation – a >= b.

a=9

b=8

operator.ge(a,b)
# ne(a,b) :- This function is used to check if a is not equal to b or is equal. 

# Returns true if a is not equal to b, else returns false. Operation – a != b.

a=4

b=5

operator.ne(a,b)
F = 'Dhananjay'

M = 'Ravindra'

L = 'Mali'



F+M+L

F+' '+M+' '+L
F*8
F in 'My name is Dhananjay'
L not in 'My name is Dhananjay' 
# chr()	Converts an integer to a character

# ord()	Converts a character to an integer

# len()	Returns the length of a string

# str()	Returns a string representation of an object
# String Indexing



s='I am a string'

s[9]
s.replace(' ','|').split('|')
# capitalize() : returns a copy of s with the first character converted to uppercase and all other characters converted to lowercase:

# lower() : Converts alphabetic characters to lowercase.

# swapcase() : returns a copy of s with uppercase alphabetic characters converted to lowercase and vice versa:

# title() : returns a copy of s in which the first letter of each word is converted to uppercase and remaining letters are lowercase:

# upper() : returns a copy of s with all alphabetic characters converted to uppercase:

# count(<sub>) : returns the number of non-overlapping occurrences of substring <sub> in string:

# endswith(<suffix>) : returns True if string ends with the specified <suffix> and False otherwise:

# find() : to see if a Python string contains a particular substring. s.find(<sub>) returns the lowest index in s where substring <sub> is found:

 

# Link : https://realpython.com/python-strings/ 

s='mY nAmE Is RaHUl'

s.capitalize()
#Averages and measures of central location: 

# mean() : Arithmetic mean (“average”) of data.

# harmonic_mean() : Harmonic mean of data.

# median() : Median (middle value) of data.

# mode() : Mode (most common value) of discrete data.



# pstdev() : Population standard deviation of data.

# pvariance() : Population variance of data.

# stdev() : Sample standard deviation of data.

# variance() : Sample variance of data.





# Link : 1) https://www.geeksforgeeks.org/python-statistics-mean-function/

#        2) https://docs.python.org/3/library/statistics.html



# Decimal Functions : 1) sqrt()

#                     2) exp()

#                     3) ln()

#                     4) log10()

#                     5) as_tuple()

#                     6) fma()

#                     7) compare()

#                     8) compare_total_mag()

#                     9) copy_abs()

#                     10) copy_negate()

#                     11) copy_sign()

#                     12) max()

#                     13) min()
import decimal

decimal.Decimal(4).sqrt()
decimal.Decimal(4).exp()
decimal.Decimal(4).ln()
decimal.Decimal(4).log10()
decimal.Decimal(-4.6).as_tuple()
decimal.Decimal(5).fma(2,20)
a=decimal.Decimal(4.9)

b=decimal.Decimal(-9.2)

a.compare(b)
a=decimal.Decimal(4.9)

b=decimal.Decimal(-9.2)

a.compare_total_mag(b)
decimal.Decimal(-9.6).copy_abs()
decimal.Decimal(-9.5).copy_negate()
a=decimal.Decimal(9.5)

b=decimal.Decimal(-2)

a.copy_sign(b)
a=decimal.Decimal(9.5)

b=decimal.Decimal(-2)

a.min(b)
a=decimal.Decimal(9.5)

b=decimal.Decimal(-2)

a.max(b)
# Iterator Functions : 1) accumulate()

#                      2) chain()

#                      3) chain.from_iterable()

#                      4) compress()

#                      5) dropwhile()

#                      6) filterfalse()