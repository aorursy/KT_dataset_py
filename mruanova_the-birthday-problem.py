people = 23
import numpy as np
from scipy import stats
year = np.zeros(366, dtype=int) # including February 29th
results = np.array([])
for i in range(people):
    birthday = np.random.randint(0, 366, 1) # one number from 0 to 366
    year[birthday] = year[birthday] + 1
print(year)
matches = 0
for i in range(366):
    if(year[i]>1):
        matches = matches + year[i]
print('people: {}'.format(people))
print('matches: {}'.format(matches))
import math, random, sys
from decimal import Decimal as dml
expected =  (1 - dml(math.factorial(365)) / ( 365**people * math.factorial(365-people)))*100
expected = np.floor(expected)
expected = int(expected)
print('expected: {} %'.format(expected))
people = 47
import numpy as np
from scipy import stats
year = np.zeros(366, dtype=int) # including February 29th
results = np.array([])
for i in range(people):
    birthday = np.random.randint(0, 366, 1) # one number from 0 to 366
    year[birthday] = year[birthday] + 1
print(year)
matches = 0
for i in range(366):
    if(year[i]>1):
        matches = matches + year[i]
print('people: {}'.format(people))
print('matches: {}'.format(matches))
import math, random, sys
from decimal import Decimal as dml
expected =  (1 - dml(math.factorial(365)) / ( 365**people * math.factorial(365-people)))*100
expected = np.floor(expected)
expected = int(expected)
print('expected: {} %'.format(expected))
people = 59
import numpy as np
from scipy import stats
year = np.zeros(366, dtype=int) # including February 29th
results = np.array([])
for i in range(people):
    birthday = np.random.randint(0, 366, 1) # one number from 0 to 366
    year[birthday] = year[birthday] + 1
print(year)
matches = 0
for i in range(366):
    if(year[i]>1):
        matches = matches + year[i]
print('people: {}'.format(people))
print('matches: {}'.format(matches))
import math, random, sys
from decimal import Decimal as dml
expected =  (1 - dml(math.factorial(365)) / ( 365**people * math.factorial(365-people)))*100
expected = np.floor(expected)
expected = int(expected)
print('expected: {} %'.format(expected))