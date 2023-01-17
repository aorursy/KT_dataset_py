import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

d = {'Apples': [30], 'Bananas': [21]}

fruits = pd.DataFrame(d)



q1.check()

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

d = {'Apples': [35, 41], 'Bananas': [21, 34]}

fruit_sales = pd.DataFrame(d)

fruit_sales.rename(index={0:'2017 Sales',1:'2018 Sales'},inplace=True)



q2.check()

fruit_sales
values = ['4 cups', '1 cup', '2 large', '1 can']

keys = ['Flour', 'Milk', 'Eggs', 'Spam']

ingredients = pd.Series(values, index=keys, name='Dinner')



q3.check()

ingredients
file_location = "../input/wine-reviews/winemag-data_first150k.csv"

reviews =  pd.read_csv(file_location, index_col=0)

#Option --> Playing with drop

#reviews.drop("Unnamed: 0", axis=1)

#Unnamed doesn't work but other reasonable names would



q4.check()

reviews
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here

animals.to_csv(path_or_buf='cows_and_goats.csv')

#Look at documentation!

# Save to csv from Pandas

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

# search for name... first one works!

q5.check()
#q5.hint()

#q5.solution()
import sqlite3

conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")

music_reviews = pd.read_sql_query("SELECT * FROM artists", conn)



q6.check()

music_reviews