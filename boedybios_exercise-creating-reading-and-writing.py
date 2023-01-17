import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])



q1.check()

fruits
# q1.hint()

# q1.solution()
fruit_sales = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'],

                index=['2017 Sales', '2018 Sales'])



q2.check()

fruit_sales
# q2.hint()

# q2.solution()
quantities = ['4 cups', '1 cup', '2 large', '1 can']

items = ['Flour', 'Milk', 'Eggs', 'Spam']

recipe = pd.Series(quantities, index=items, name='Dinner')



q3.check()

recipe
# q3.hint()

# q3.solution()
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)



q4.check()

reviews
# q4.hint()

# q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
animals.to_csv("cows_and_goats.csv")



q5.check()
# q5.hint()

# q5.solution()