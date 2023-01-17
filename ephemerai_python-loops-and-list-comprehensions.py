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
for i in range(5):
    print("Doing important work. i =", i)
r = range(5)
r
help(range)
list(range(5))
nums = [1, 2, 4, 8, 16]
for i in range(len(nums)):
    nums[i] = nums[i] * 2
nums
def double_odds(nums):
    for i, num in enumerate(nums):
        if num % 2 == 1:
            nums[i] = num * 2

x = list(range(10))
double_odds(x)
x
list(enumerate(['a', 'b']))
x = 0.125
numerator, denominator = x.as_integer_ratio()
print(numerator, denominator)
nums = [
    ('one', 1, 'I'),
    ('two', 2, 'II'),
    ('three', 3, 'III'),
    ('four', 4, 'IV'),
]

for word, integer, roman_numeral in nums:
    print(integer, word, roman_numeral, sep=' = ', end='; ')
for tup in nums:
    word = tup[0]
    integer = tup[1]
    roman_numeral = tup[2]
    print(integer, word, roman_numeral, sep=' = ', end='; ')
i = 0
while i < 10:
    print(i, end=' ')
    i += 1
squares = [n**2 for n in range(10)]
squares
squares = []
for n in range(10):
    squares.append(n**2)
squares
short_planets = [planet for planet in planets if len(planet) < 6]
short_planets
# str.upper() returns an all-caps version of a string
loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
loud_short_planets
[
    planet.upper() + '!' 
    for planet in planets 
    if len(planet) < 6
]
[32 for planet in planets]
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    n_negative = 0
    for num in nums:
        if num < 0:
            n_negative = n_negative + 1
    return n_negative

count_negatives([5, -1, -2, 0, 3])
def count_negatives(nums):
    return len([num for num in nums if num < 0])

count_negatives([5, -1, -2, 0, 3])
def count_negatives(nums):
    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of 
    # Python where it calculates something like True + True + False + True to be equal to 3.
    return sum([num < 0 for num in nums])

count_negatives([5, -1, -2, 0, 3])