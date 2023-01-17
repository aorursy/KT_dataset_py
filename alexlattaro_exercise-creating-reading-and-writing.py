import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")

fruits = pd.DataFrame ({'Apples':[30], 'Bananas':[21]})





q1.check()

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

fruit_sales = pd.DataFrame ({'2017 Sales':{'Apples': 35,'Bananas':21},

                            '2018 Sales':{'Apples':41, 'Bananas': 34}}).T



q2.check()

fruit_sales
#q2.hint()

#q2.solution()
ingredients =  pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=["Flour", "Milk", "Eggs", "Spam"], name="Dinner")

q3.check()

ingredients
#q3.hint()

#q3.solution()
reviews = pd.read_csv ('../input/wine-reviews/winemag-data_first150k.csv')

reviews = reviews.drop('Unnamed: 0', axis =1)



q4.check()

reviews
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals

animals.to_csv("cows_and_goats.csv")



q5.check()
#q5.hint()

#q5.solution()