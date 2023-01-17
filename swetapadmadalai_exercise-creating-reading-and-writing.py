import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

import pandas as pd

fruits = pd.DataFrame({"Apples": [30],"Bananas": [21]})



# Check your answer



fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

import pandas as pd

fruit_sales = pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])



# Check your answer



fruit_sales
#q2.hint()

#q2.solution()
ingredients = pd.Series(["4 cups","1 cup","2 Large","1 can"],index=["Flour","Milk","Eggs","Spam"],name='Dinner')



# Check your answer



ingredients
#q3.hint()

#q3.solution()
import pandas as pd

wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

wine_reviews.head()
import pandas as pd

reviews =pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



# Check your answer



reviews
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here

import pandas as pd 

cows_and_goats=pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])



x=df.to_csv('cows_and_goats',encoding='utf-8')

x

# Check your answer

#q5.check()
#q5.hint()

#q5.solution()