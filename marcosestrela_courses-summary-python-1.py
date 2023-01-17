# Variable assignment

variable = 10



# Reassigning variable

variable = variable + 10



# Function calls

print(variable)
# Describe the type of "thing" that variable is

print(type(variable)) # int

print(type(19.95))    # float
# True division

print(5 / 2) # 2.5

print(6 / 2) # 3.0



# Floor division

print(5 // 2) # 2

print(6 // 2) # 3



# Modulus

print(5 % 2) # 1

print(6 % 2) # 0
# Minimun value

print("Min:", min(1, 2, 3))  # 1



# Maximun value

print("Max:", max(1, 2, 3))  # 3



# Absolute value

print("Abs:", abs(-32))     # 32



# Cast to float and to integer

print("Float:",float(10))    # 10.0

print("Integer:",int(3.33))    # 3

print("Integer:",int("807"))   # 807

help(print) # Common pitfall: pass in the name of the function itself, and not the result of calling.
def defining_function(a, b = None): # Expect obrigatory parameter and optional parameter

    """Docstring that is return when the user calls the help functiom    

    >>> defining_function(a)

    a

    """

    return a # Return is optional



# You can supply functions as arguments to other functions.

defining_function(defining_function(10))
x, y = True, False

print(x,y)

print(type(x), type(y))
print(True and True)  # and

print(True or False)  # or

print(not True)       # not

# Precedence: not, and, or
x = 2

if(x == 0):               # if

  print(x, "is zero")

elif(x < 1):

  print(x, "is negative") # elif

else:

  print(x, "is positive") # else
print(bool(3), 3)                                 # All numbers are treated as true, except 0 

print(bool(0), 0 )

print(bool("test"), "test")                       # All strings are treated as true, except the empty string ""

print(bool([]), "[]")                             # Generally empty sequences are "falsey" and the rest are "truthy"    

print(bool({'dict': 'value'}), {'dict': 'value'})

expression = True

variable = 'True' if expression else 'False'

print(variable)