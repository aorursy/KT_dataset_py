import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
print(answer_q1())
# Your code here
d = {'Apples': [30] , 'Bananas': [21] }
df = pd.DataFrame(data=d)
df
# Your code here
d = {'Apples': [35,41] , 'Bananas': [21,34] }
df2 = pd.DataFrame(data=d).rename({0:'2017 Slaes',1:'2018 Slaes'},axis="index")
df2
# Your code 
series = pd.Series(data=['4 cups','1 cup', '2 large', '1 can'], index=['Flour','Milk','Eggs','Spam'],name='Dinner')
series
# Your code here 
input_df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
input_df.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')

# Your Code Here
import sqlite3
cnx = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df3 = pd.read_sql_query('SELECT * FROM artists',cnx)
df3.head()
