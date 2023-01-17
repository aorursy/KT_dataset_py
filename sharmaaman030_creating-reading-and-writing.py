import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

fruits=pd.DataFrame(data=[[30,21]],columns=['Apples','Bananas'])



q1.check()

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

fruits={'Apples':[35,41],

       'Bananas':[21,34]}

df=pd.DataFrame(fruits,columns=['Apples','Bananas'])

s=pd.Series(['2017 Sales','2018 Sales'])

fruit_sales=df.set_index([s])

print(fruit_sales)

q2.check()

fruit_sales
#q2.hint()

#q2.solution()
I=['Flour','Milk','Eggs','Spam']

O=['4 cups','1 cup','2 large','1 can']



ingredients =pd.Series(O,I,name="Dinner")

print(ingredients)





q3.check()

#q3.hint()

#q3.solution()
url='../input/wine-reviews/winemag-data_first150k.csv'

reviews =pd.read_csv(url,index_col=0)



reviews

q4.check()
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals.to_csv('cows_and_goats.csv')



q5.check()
#q5.hint()

#q5.solution()