import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

fruits = pd.DataFrame({'Apples':[30],'Bananas':[21]})



# Check your answer

q1.check()

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

fruit_sales = pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]}, index = ['2017 Sales','2018 Sales'])



# Check your answer

q2.check()

fruit_sales
#q2.hint()

#q2.solution()
ingredients = pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name='Dinner')



# Check your answer

q3.check()

ingredients
#q3.hint()

#q3.solution()
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)



# Check your answer

q4.check()

reviews
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here

animals.to_csv("cows_and_goats.csv")

# Check your answer

q5.check()
#q5.hint()

#q5.solution()
reviews.head()
reviews.shape
reviews.describe
reviews.tail()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(25,20))

plt.title("comparison of wine prices in different countries")





sns.barplot(x = reviews['country'],y=reviews['price'])

plt.xlabel("Different types of countries")

plt.ylabel("Price variation of wines")
reviews.columns
kepler = pd.read_csv('../input/kepler-exoplanet-search-results/cumulative.csv')
print(kepler)
kepler
kepler.describe()
kepler.head()
kepler.tail()
chess = pd.read_csv('../input/chess/games.csv')
chess
chess.head()
ramen = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv',index_col='Brand')
ramen
ramen.describe()

ramen.head()
plt.figure(figsize=(25,20))

plt.title("comparison of different brands of Ramen in various countries")





sns.barplot(x = reviews['country'],y=reviews['price'])

plt.xlabel("Name of the country")

plt.ylabel("Star rating of Ramen")
plt.figure(figsize=(14,7))

plt.title("comparison of different brands of Ramen in various countries")





sns.regplot(x = reviews['points'],y=reviews['price'])

plt.xlabel("Points")

plt.ylabel("Price of wine")
plt.figure(figsize=(14,7))

plt.title("comparison of different brands of Ramen in various countries")





sns.scatterplot(x = reviews['points'],y=reviews['price'])

plt.xlabel("Points")

plt.ylabel("Price of wine")