def least_difference(a, b, c):
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1,diff2,diff3)
print(
    least_difference(1, 10, 100),
    least_difference(1, 10, 10),
    least_difference(5, 6, 7), # Python allows trailing commas in argument lists. How nice is that?
)
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
def can_run_for_president(age):
    """Can someone of the given age run for president in the US?"""
    # The US Constitution says you must "have attained to the Age of thirty-five Years"
    return age >= 35

print("Can a 19-year-old run for president?", can_run_for_president(19))
print("Can a 45-year-old run for president?", can_run_for_president(45))
def can_run_for_president(age, is_natural_born_citizen):
    """Can someone of the given age and citizenship status run for president in the US?"""
    # The US Constitution says you must be a natural born citizen *and* at least 35 years old
    return is_natural_born_citizen and (age >= 35)

print(can_run_for_president(19, True))
print(can_run_for_president(55, False))
print(can_run_for_president(55, True))
prepared_for_weather = (
    have_umbrella 
    or ((rain_level < 5) and have_hood) 
    or (not (rain_level > 0 and is_workday))
)
def inspect(x):
    if x == 0:
        print(x, "is zero")
    elif x > 0:
        print(x, "is positive")
    elif x < 0:
        print(x, "is negative")
    else:
        print(x, "is unlike anything I've ever seen...")

inspect(0)
inspect(-15)
def f(x):
    if x > 0:
        print("Only printed when x is positive; x =", x)
        print("Also only printed when x is positive; x =", x)
    print("Always printed, regardless of x's value; x =", x)

f(1)
f(0)
def quiz_message(grade):
    if grade < 50:
        outcome = 'failed'
    else:
        outcome = 'passed'
    print('You', outcome, 'the quiz with a grade of', grade)
    
quiz_message(80)
def quiz_message(grade):
    outcome = 'failed' if grade < 50 else 'passed'
    print('You', outcome, 'the quiz with a grade of', grade)
    
quiz_message(45)
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planets[0:3]
planets[:3]

planets[3:]
x = 12
# x is a real number, so its imaginary part is 0.
print(x.imag)
# Here's how to make a complex number, in case you've ever been curious:
c = 12 + 3j
print(c.imag)
x.bit_length
# To actually call it, we add parentheses:
x.bit_length() 
planets.append('Pluto')
#removes and returns the last element of a list
planets.pop() 
planets.index('Earth') # 0-indexing
# Is Earth a planet?
"Earth" in planets
#1: The syntax for creating them uses (optional) parentheses rather than square brackets
t = (1, 2, 3)
t = 1, 2, 3 # equivalent to above
# 2: They cannot be modified (they are immutable).
t[0] = 100
# For example, the as_integer_ratio() method of float objects returns a numerator and a denominator in the form of a tuple:
x = 0.125
x.as_integer_ratio()
# These multiple return values can be individually assigned as follows:
numerator, denominator = x.as_integer_ratio()
print(numerator / denominator)
# Finally , swapping two variables!
a = 1
b = 0
a, b = b, a
print(a, b)
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ') # print all on same line
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
product
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')  
# Note that the range starts at zero, 
# and that by convention the top of the range is not included in the output. 
# range(5) gives the numbers from 0 up to but not including 5.
for i in range(5):
    print("Doing important work. i =", i)
nums = [1, 2, 4, 8, 16]
for i in range(len(nums)):
    nums[i] = nums[i] * 2
nums
# Wrong way to iterate 
nums = [1, 2, 4, 8, 16]
for i in nums: 
    nums[i] = nums[i] * 2
nums
nums = [1, 2, 4, 8, 16]
prod=1
for i in nums: 
    prod = prod * 2
prod




