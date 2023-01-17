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
print(planet)

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']





for planet in planets:

    print(planet, end=', ')



multiplicands = (2, 2, 2, 3, 3, 5)

product = 1

for mult in multiplicands:

    product = product * mult

print(product)
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'

# print all the uppercase letters in s, one at a time

for char in s:

    if char.isupper():

        print(char, end=' ')      
for i in range(5):

    print("Doing important work. i =", i)
i = 0

while i < 10:

    i += 1

    print(i, end=' ')

    i += 1


squares = [n**2 for n in range(10)]

print(squares)
# str.upper() returns an all-caps version of a string

loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]

print(loud_short_planets)
s = [planet.upper() + '!' for planet in planets]

print(s)