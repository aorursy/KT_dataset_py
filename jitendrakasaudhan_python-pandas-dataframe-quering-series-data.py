import pandas as pd

# pd.Series? # returns info page for Series object
animals = ['Rat', 'Cat', 'Dog']

pd.Series(animals)
nums = [1,2,3,4]

pd.Series(nums)
# if there is None in a list, pandas will create object of type None

animals = ['Rat', 'Cat', None]

pd.Series(animals)

nums = [2,3,4,None]

pd.Series(nums)
import numpy as np

np.nan == np.nan # is not a number is not equal to itself, we should np.isnan function to check if it is nan

np.isnan(np.nan)
np.nan == None # np.nan and None are not equal
sports = {'Archery': 'Bhutan',

          'Golf': 'Scotland',

          'Sumo': 'Japan',

          'Taekwondo': 'South Korea'}

sp = pd.Series(sports)
sp.iloc[2] # i means index loc means location, if we want to get the value using index than use .iloc which is an attribute of a series
sp.loc['Golf']
sp.index
sp[3] # we can query series wihout using .iloc or .loc but using .iloc or .loc is efficient than not using it
int_series = pd.Series([100,200,300,400,500,600])

int_series[0]
sports = {99: 'Bhutan',

          100: 'Scotland',

          101: 'Japan',

          102: 'South Korea'}

s = pd.Series(sports)
s[0]#This won't call s.iloc[0] as one might expect, it generates an error instead
s = pd.Series([100.00, 120.00, 101.00, 3.00])

print(s)

print(s+2)  #adds two to each item in s using broadcasting



# above can be done using series interables but its not efficient

import numpy as np

ra = pd.Series(np.random.randint(0, 1000, 10000))

print(ra.head())





for label, value in ra.iteritems():

    ra.loc[label] = value+2

    

print(ra.head(4))
total = 0

for num in s:

    total += num

    

print(total)
%%timeit -n 100

total = 0

for num in s:

    total += num

    

import numpy as np

rand = pd.Series(np.random.randint(0,1000, 10000))

rand.head(4)
len(rand)
%%timeit -n 100 # calculate time consumed for 100 runs

sum = 0

for num in rand:

    sum+=num
%%timeit -n 100

sum = np.sum(rand)
original_sports = pd.Series({'Archery': 'Bhutan',

                             'Golf': 'Scotland',

                             'Sumo': 'Japan',

                             'Taekwondo': 'South Korea'})

cricket_loving_countries = pd.Series(['Australia',

                                      'Barbados',

                                      'Pakistan',

                                      'England'], 

                                   index=['Cricket',

                                          'Cricket',

                                          'Cricket',

                                          'Cricket'])

all_sports = original_sports.append(cricket_loving_countries)

print(original_sports) # does not change base object

print(all_sports)
import pandas as pd

purchase_1 = pd.Series({'Name': 'Chris',

                        'Item Purchased': 'Dog Food',

                        'Cost': 22.50})

purchase_2 = pd.Series({'Name': 'Kevyn',

                        'Item Purchased': 'Kitty Litter',

                        'Cost': 2.50})

purchase_3 = pd.Series({'Name': 'Vinod',

                        'Item Purchased': 'Bird Seed',

                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 2', 'Store 3'])

df
df.loc['Store 1'] # get data by index
df.loc[:, 'Cost'] # select all rows and Cost column