10 + 5
10 - 5
10 * 5
10 / 3
10 // 3
10 ** 3
# These operations are executed in reverse order of appearance due to the order of operations.



2 + 3 * 5 ** 2
# This time, the addition comes first and the exponentiation comes last.



((2 + 3) * 5 ) ** 2
100 % 75
import math     # Load the math module
# math.log() takes the natural logarithm of its argument:



math.log(2.7182)
# Add a second argument to specify the log base:



math.log(100, 10)       # Take the log base 10 of 100
# math.exp() raises e to the power of its argument



math.exp(10) 
# Use math.sqrt() to take the square root of a number:



math.sqrt(64)
# Use abs() to get the absolute value of a number. Note abs() is a base Python function so you do not need to load the math package to use it.



abs(-30)
math.pi   # Get the constant pi
# Use round() to round a number to the nearest whole number:



round(233.234)
# Add a second argument to round to a specified decimal place



round(233.234, 1)   # round to 1 decimal place
# Enter a negative number to round to the left of the decimal



round(233.234, -1)   # round to the 10's place
# Round down to the nearest whole number with math.floor()



math.floor(2.8) 
# Round up with math.ciel()



math.ceil(2.2)