import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1(df)

check_q2(pd.DataFrame({'Apples' : [35, 41], 'Bananas' : [21, 34]}, index=['2017 Sales', '2018 Sales']))
# Your code here
s = pd.Series({'Flour' : '4 cups', 'Milk' : '1 cup', 'Eggs' : '2 large', 'Spam' : '1 can'}, name='Dinner')
s
check_q3(pd.Series({'Flour' : '4 cups', 'Milk' : '1 cup', 'Eggs' : '2 large', 'Spam' : '1 can'}, name='Dinner'))



# Your code here
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
check_q4(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
import sqlite3

conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query('select * from artists;', conn)
check_q7(df)