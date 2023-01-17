import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame([[30, 21]],columns=['Apples','Bananas'])

q1.check()
fruits
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame([[35,21],[41, 34]], index=['2017 Sales', '2018 Sales'], columns=['Apples', 'Bananas'])

q2.check()
fruit_sales
ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour', 'Milk', 'Eggs','Spam'], name = 'Dinner')

q3.check()
ingredients
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)

q4.check()
reviews
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
# Your code goes here
animals.to_csv('cows_and_goats.csv')
q5.check()
#q5.hint()
#q5.solution()
import sqlite3
music_reviews = pd.read_sql_query("SELECT * FROM artists", sqlite3.connect('../input/pitchfork-data/database.sqlite')) 

q6.check()
music_reviews