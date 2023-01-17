import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
#fruits = ____
fruit = {'Apples': [30], 'Bananas': [21]}
fruits = pd.DataFrame(fruit, columns = ['Apples', 'Bananas'])
q1.check()
fruits
# Uncomment the line below to see a solution
#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
sales_index = ['2017 Sales','2018 Sales']
fruit_sales = pd.DataFrame([[35, 21],[41, 34]], index=sales_index, columns=['Apples', 'Bananas'])
q2.check()
fruit_sales
#q2.solution()
ingredients = pd.Series(['4 cups','1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], 
                        name= 'Dinner', dtype='object')

q3.check()
ingredients
#q3.solution()

filepath = '../input/wine-reviews/winemag-data_first150k.csv'
reviews = pd.read_csv(filepath, index_col=0)

q4.check()
reviews
#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
# Your code goes here
animals.to_csv('cows_and_goats.csv')
q5.check()
#q5.solution()
import sqlite3
filepath = '../input/pitchfork-data/database.sqlite'
dbconnect = sqlite3.connect(filepath)
music_reviews = pd.read_sql_query("SELECT * FROM artists", dbconnect)

q6.check()
music_reviews
#q6.solution()