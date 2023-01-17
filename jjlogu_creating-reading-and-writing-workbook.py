import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

data={"Apples":[30], "Bananas": [21] }

fruits = pd.DataFrame(data)



q1.check()

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

data = {"Apples":[35,41], "Bananas": [21,34] }

fruit_sales = pd.DataFrame(data,index=["2017 Sales", "2018 Sales"])



q2.check()

fruit_sales
q2.hint()

#q2.solution()
data = ["4 cups", "1 cup", "2 large", "1 can"]

ingredients = pd.Series(data, index = ["Flour", "Milk", "Eggs", "Spam"], name="Dinner")



q3.check()

ingredients
#q3.hint()

#q3.solution()
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)



q4.check()

reviews
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here

animals.to_csv("cows_and_goats.csv")



q5.check()
#q5.hint()

#q5.solution()
from sqlalchemy import create_engine

engine = create_engine("sqlite:///../input/pitchfork-data/database.sqlite")

music_reviews = pd.read_sql("select * from artists", engine)



q6.check()

music_reviews
#q6.hint()

#q6.solution()