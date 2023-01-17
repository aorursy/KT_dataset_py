print("The print function takes an input and prints it to the screen.")
print("Each call to print starts on a new line.")
print("You'll often call print with strings, but you can pass any kind of value. For example, a number:")
print(2 + 2)
print("If print is called with multiple arguments...", "it joins them",
      "(with spaces in between)", "before printing.")
print('But', 'this', 'is', 'configurable', sep='!...')
print()
print("^^^ print can also be called with no arguments to print a blank line.")
help(abs)
help(abs(-2))
help(print)
def least_difference(a, b, c):
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)
print(
    least_difference(1, 10, 100),
    least_difference(1, 10, 10),
    least_difference(5, 6, 7), # Python allows trailing commas in argument lists. How nice is that?
)
help(least_difference)
def least_difference(a, b, c):
    """Return the smallest difference between any two numbers
    among a, b and c.
    
    >>> least_difference(1, 5, -5)
    4
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)
help(least_difference)
def least_difference(a, b, c):
    """Return the smallest difference between any two numbers
    among a, b and c.
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    min(diff1, diff2, diff3)
    
print(
    least_difference(1, 10, 100),
    least_difference(1, 10, 10),
    least_difference(5, 6, 7),
)
mystery = print()
print(mystery)
print(1, 2, 3, sep=' < ')
print(1, 2, 3, sep='')
print(1, 2, 3)
def greet(who="Colin"):
    print("Hello,", who)
    
greet()
greet(who="Kaggle")
# (In this case, we don't need to specify the name of the argument, because it's unambiguous.)
greet("world")
def f(n):
    return n * 2

x = 12.5
print(
    type(x),
    type(f), sep='\n'
)
print(x)
print(f)
def call(fn, arg):
    """Call fn on arg"""
    return fn(arg)

def squared_call(fn, arg):
    """Call fn on the result of calling fn on arg"""
    return fn(fn(arg))

print(
    call(f, 1),
    squared_call(f, 1), 
    sep='\n', # '\n' is the newline character - it starts a new line
)
def mod_5(x):
    """Return the remainder of x after dividing by 5"""
    return x % 5

print(
    'Which number is biggest?',
    max(100, 51, 14),
    'Which number is the biggest modulo 5?',
    max(100, 51, 14, key=mod_5),
    sep='\n',
)
mod_5 = lambda x: x % 5

# Note that we don't use the "return" keyword above (it's implicit)
# (The line below would produce a SyntaxError)
#mod_5 = lambda x: return x % 5

print('101 mod 5 =', mod_5(101))
# Lambdas can take multiple comma-separated arguments
abs_diff = lambda a, b: abs(a-b)
print("Absolute difference of 5 and 7 is", abs_diff(5, 7))
# Or no arguments
always_32 = lambda: 32
always_32()
# Preview of lists and strings. (We'll go in depth into both soon)
# - len: return the length of a sequence (such as a string or list)
# - sorted: return a sorted version of the given sequence (optional key 
#           function works similarly to max and min)
# - s.lower() : return a lowercase version of string s
names = ['jacques', 'Ty', 'Mia', 'pui-wa']
print("Longest name is:", max(names, key=lambda name: len(name))) # or just key=len
print("Names sorted case insensitive:", sorted(names, key=lambda name: name.lower()))