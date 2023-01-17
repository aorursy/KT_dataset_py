# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
hello = "hello\nworld"

print(hello)

triplequoted_hello = """hello

world"""

print(triplequoted_hello)



print(triplequoted_hello == hello)
planet = 'Pluto'

print(planet[0])

print(planet[-3:])



# Yes, we can even loop over them

print([char + "!"  for char in planet])



claim = "Pluto is a planet!"

claim.upper()

claim.startswith("Pluto")
datestr = '1956-01-31'

year, month, day = datestr.split('-')

print(year)



print("-".join([month, day, year]))





words = claim.split()



print(words)



# Yes, we can put unicode characters right in our string literals :)

' 👏 '.join([word.upper() for word in words])
position = 9

"{}, you'll always be the {}th planet to me.".format(planet, position)

pluto_mass = 1.303 * 10**22

earth_mass = 5.9722 * 10**24

population = 52910390

#         2 decimal points   3 decimal points, format as percent     separate with commas

"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(

    planet, pluto_mass, pluto_mass / earth_mass, population,

)
# Referring to format() arguments by index, starting from 0

s = """Pluto's a {0}.

No, it's a {1}.

{0}!

{1}!""".format('planet', 'dwarf planet')

print(s)
numbers = {'one':1, 'two':2, 'three':3}

numbers['one']
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planet_to_initial = {planet: planet[0] for planet in planets}

planet_to_initial
# A for loop over a dictionary will loop over its keys

for k in numbers:

    print("{} = {}".format(k, numbers[k]))
# Get all the initials, sort them alphabetically, and put them in a space-separated string.

' '.join(sorted(planet_to_initial.values()))
' '.join(sorted(planet_to_initial.keys()))
for planet, initial in planet_to_initial.items():

    print("{} begins with \"{}\"".format(planet.rjust(10), initial))



planetkk = {planet: planet[0] for planet in planets}

planetkk
planet_to_initial.items()