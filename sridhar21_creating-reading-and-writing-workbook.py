import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
df = pd.DataFrame({'Apples': [30], 'Bananas' : [21]})
print(check_q1(df))
df = pd.DataFrame({'Apples' : [35, 41], 'Bananas' : [21, 34]}, index= ['2017 Sales', '2018 Sales'])
print(check_q2(df))
arr = ['4 cups', '1 cup', '2 large', '1 can']
df = pd.Series(arr, index= ['Flour', 'Milk', 'Eggs', 'Spam'], name= 'Dinner')
print(check_q3(df))
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
print(check_q4(df))

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
print(check_q6(df))
import sqlite3

conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query("SELECT * FROM artists", conn)
print(check_q7(df))