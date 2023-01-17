import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits_dict=[{'Apples':30,'Bananas':21}]
fruits = pd.DataFrame(fruits_dict)
fruits.head()

q1.check()
fruits
# Uncomment the line below to see a solution
#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
alist = [('2017 Sales',35,21),('2018 Sales', 41,34)]
columns=['','Apples','Bananas']
fruit_sales=pd.DataFrame(alist, columns=columns).set_index('')


q2.check()
fruit_sales
#q2.solution()
ingredients = {'Flour': '4 cups', 'Milk' : '1 cup', 'Eggs' : '2 large', 'Spam': '1 can'}
ingredients= pd.Series(ingredients, name='Dinner')

q3.check()
ingredients
#q3.solution()
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv').set_index(reviews.columns[0])

q4.check()
reviews
#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
animals.to_csv('cows_and_goats.csv')# Your code goes here

q5.check()
#q5.solution()
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
music_reviews = pd.read_sql_query("SELECT * FROM artists", conn)

q6.check()
music_reviews
#q6.solution()