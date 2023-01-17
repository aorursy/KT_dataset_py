import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")
# Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame({ 'Apples': [30], 'Bananas': [21]})
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])

fruits
# Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'],
                index=['2017 Sales', '2018 Sales'])
fruit_sales
quantities = ['4 cups', '1 cup', '2 large', '1 can']
indexes = ['Flour', 'Milk', 'Eggs', 'Spam']
ingredients = pd.Series(quantities, index=indexes, name='Dinner')
ingredients

reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
reviews
reviews.head()
reviews.tail()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
animals.to_csv("cows_and_goats.csv")