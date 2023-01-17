import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

import pandas as pd

fruits = pd.DataFrame({"Apples" : [30],"Bananas" : [21]},index = [0])

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

import pandas as pd

fruit_sales=pd.DataFrame({"Apples":[35,41],"Bananas":[21,34]},index=["2017 Sales","2018 Sales"])

fruit_sales
#q2.hint()

#q2.solution()
ingredients = ____



q3.check()

ingredients
#q3.hint()

#q3.solution()
reviews = ____



q4.check()

reviews
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here

import pandas as pd

df=pd.read_csv(cows_and_goats.csv)

df.to_csv(cows_and_goats.csv)
#q5.hint()

#q5.solution()
music_reviews = ____



q6.check()

music_reviews
#q6.hint()

#q6.solution()