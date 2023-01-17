from learntools.core import binder; binder.bind(globals())
from learntools.python.ex4 import *
print('Setup complete.')
primes = [2, 3, 5, 7]
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]
# (I could also have written this on one line, but it can get hard to read)
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]
my_favourite_things = [32, 'raindrops on roses', help]
# (Yes, Python's help function is *definitely* one of my favourite things)
planets[0]
planets[-1]
planets[-2]
def select_second(L):
    """返回给定列表的第二个元素. 如果没有，返回 None.
    """
    pass
q1.check()
#q1.hint()
#q1.solution()
planets[0:3]
planets[:3]
planets[3:]
# 除了第一个和最后一个的所有元素
planets[1:-1]
# 最后三个元素
planets[-3:]
a = [1, 2, 3]
b = [1, [2, 3]]
c = []
d = [1, 2, 3][1:]

# 将预测放在下面的列表中。应包含4个数字，第一个是A的长度，第二个是B的长度，依此类推。
lengths = []

q4.check()
# 这行会提供解释
#q4.solution()
planets[3] = 'Malacandra'
planets
planets[:3] = ['Mur', 'Vee', 'Ur']
print(planets)
# (Okay, that was rather silly. Let's give them back their old names)
planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars',]
def purple_shell(racers):
    """参数是选手名单, 把第一名选手排到最后，最后一名放到第一个。
    
    >>> r = ["Mario", "Bowser", "Luigi"]
    >>> purple_shell(r)
    >>> r
    ["Luigi", "Bowser", "Mario"]
    """
    pass

q3.check()
#q3.hint()
#q3.solution()
# How many planets are there?
len(planets)
# The planets sorted in alphabetical order
sorted(planets)
primes = [2, 3, 5, 7]
sum(primes)
max(primes)
x = 12
# x is a real number, so its imaginary part is 0.
print(x.imag)
# Here's how to make a complex number, in case you've ever been curious:
c = 12 + 3j
print(c.imag)
x.bit_length
x.bit_length()
help(x.bit_length)
# Pluto is a planet darn it!
planets.append('Pluto')
help(planets.append)
planets
planets.pop()
planets
planets.index('Earth')
# Is Earth a planet?
"Earth" in planets
# Is Calbefraques a planet?
"Calbefraques" in planets
help(planets)
def fashionably_late(arrivals, name):
    """参数是有序的客人抵达顺序列表和名称列表, 返回客人是否满足‘时尚迟到’
    """
    pass

q5.check()
#q5.hint()
#q5.solution()
t = (1, 2, 3)
t = 1, 2, 3 # equivalent to above
t
t[0] = 100
x = 0.125
x.as_integer_ratio()
numerator, denominator = x.as_integer_ratio()
print(numerator / denominator)
a = 1
b = 0
a, b = b, a
print(a, b)