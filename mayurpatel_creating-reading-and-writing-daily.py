import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

d = {'Apples': [30], 'Bananas': [21]}

fruits = pd.DataFrame(data=d)



q1.check()

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

d = { 'Apples': {'2017 Sales': 35, '2018 Sales': 41}, 'Bananas': {'2017 Sales': 21, '2018 Sales': 34}}

fruit_sales = pd.DataFrame(data=d)



q2.check()

fruit_sales
#q2.hint()

#q2.solution()
d = {'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'}

ingredients = pd.Series(data=d, name='Dinner')



q3.check()

ingredients
#q3.hint()

#q3.solution()
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', usecols=[1,2,3,4,5,6,7,8,9,10])



q4.check()

reviews
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here

animals.to_csv('cows_and_goats.csv')

q5.check()
#q5.hint()

#q5.solution()