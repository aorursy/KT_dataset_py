primes = [2,3,5,7]
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
hands = [

    ['J', 'Q', 'K'],

    ['2', '2', '2'],

    ['6', 'A', 'K'], # (Comma after the last element is optional)

]

# (I could also have written this on one line, but it can get hard to read)

hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]




my_favourite_things = [32, 'raindrops on roses', help]


planets[0], planets[1]
planets[-3], planets[-1]
planets[0:3], 
planets[:3]
planets[3:]
planets[1:-1]
planets[-3:]
planets[3] = 'Malacandra'

planets
planets[:3] = ['Mur', 'Vee', 'Ur']

print(planets)

# That was silly. Let's give them back their old names

planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars',]
len(planets)
sorted(planets)
max(primes)
primes = [2, 3, 5, 7]

sum(primes)
x = 12

# x is a real number, so its imaginary part is 0.

print(x.imag)

# Here's how to make a complex number, in case you've ever been curious:

c = 12 + 3j

print(c.imag)
x.bit_length
planets.append('Pluto')

planets
planets.pop(), 

planets
planets.index('Earth')
planets.index('Pluto')
"Earth" in planets
"Calbefraques" in planets
t = (1, 2, 3)

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