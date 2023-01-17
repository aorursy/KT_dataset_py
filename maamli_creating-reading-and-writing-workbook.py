import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q7(pd.DataFrame())

data = {'Apples':[30], 'Bananas':[21]}

df = pd.DataFrame(data, columns=['Apples', 'Bananas'])

print (df)

data2 = [[35,21], [41,34]]

df2 = pd.DataFrame(data2, columns=['Apples', 'Bananas'], index= ['2017 Sales', '2018 Sales'])

print(df2)

# method 2

data2a = {'Apples':[35,41], 'Bananas':[21,34]}
df2a = pd.DataFrame(data2a, index=['2017 Sales', '2018 Sales'])
print (df2a)
data3 = ['4 cups', '1 cup', '2 large', '1 can']

ser3= pd.Series(data3, index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')

print(ser3)
wine_reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")

artists = pd.read_sql_query("SELECT * FROM artists", conn)

artists.head()