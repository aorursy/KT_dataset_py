import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")




# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

fruits = pd.DataFrame({'Apples':[30] , 'Bananas':[21] })



print(fruits)



# Check your answer

q1.check()



# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

fruit_sales = pd.DataFrame({'Apples':[35,41] , 'Bananas':[21,34] },index=['2017 Sales','2018 Sales'])



# Check your answer

q2.check()

fruit_sales
ingredients = pd.Series(["4 cups", "1 cup", "2 large", "1 can"], index=["Flour","Milk","Eggs","Spam"], name="Dinner")

print(ingredients)

q3.check()

ingredients

reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)

print(reviews)



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here

csv_file = animals.to_csv("cows_and_goats.csv")

# Check your answer

q5.check()
#q5.hint()

#q5.solution()