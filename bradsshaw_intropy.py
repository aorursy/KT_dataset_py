# this is a code cell

# a pound sign is used to create a comment

# this will comment out everything that follows on that line

# python code follows usual code notation, and might remind you of VBA as it uses the "dot" convention... i.e. object.do_something_to_object(paramters)
# use the radians function in the math package



angle = 90.



# option 1

import math



radians = math.radians(angle) # notice how we import the package "as-is", and refer to it by its full and proper name. Let's be lazy and give it a nickname



# option 2

import math as m



radians = m.radians(angle) # we've decided to call math "m", so we refer to it as "m.some_function". We can also import radians() directly



# option 3

from math import radians



result = radians(angle) # notice how since we have an object called radians, we need to change our variable name (otherwise the radians function will be overwritten)
l = ['brad','number',1,'teacher']

d = {'name':'brad','rank':1,'job':'teacher'}



print(l)

print(d)



for item in l:

    print(item)
d['name']
# import the pandas package for data manipulation; it's conventionally called "pd"

import pandas as pd



# import the meteorites CSV. The import statement would look different if you're working outside of a Kaggle kernel

df = pd.read_csv('../input/meteorite-landings/meteorite_landings_prod.csv')



print('Import complete!')
df.head()
# quickly demo attribute v method using data



# shape is a pandas DataFrame attribute that gives a tuple of (number of rows,number of columns)

# sum is a pandas DataFrame / Series method that we will apply to a single column



print(df.shape)

print(df['mass'].sum())
# iloc v loc

# integer v general

# excludes last entry v includes last entry



# select the mass column:

df['mass']

#df.mass

#df.loc[:,'mass']

#df.iloc[:,'mass'] #?

#df.iloc[:,7]
# select the 42nd row of mass

#df.iloc[41,7]

#df.loc[41,'mass']
# conditionally select

x = df.loc[df['location_code']=='DE',['mass','location','location_code']].copy()

x['mass'].sum()

x['mass'].mean()
# add a new column called centred mass

x['centred_mass'] = x['mass'] - x['mass'].mean()

x.head()