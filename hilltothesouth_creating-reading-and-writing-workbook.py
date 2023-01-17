import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
df1 = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1 (df1)
df2 = pd.DataFrame ({'Apples': [35, 41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])
check_q2 (df2)
s3 = pd.Series (['4 cups', '1 cup', '2 large', '1 can'],
                index = ['Flour', 'Milk', 'Eggs', 'Spam'],
                name = 'Dinner')
check_q3 (s1)
df4 = pd.read_csv ('../input/wine-reviews/winemag-data_first150k.csv')
del df4['Unnamed: 0']
check_q4 (df4)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
df6 = q6_df.to_csv ('cows_and_goats.csv')
check_q6 (df6)
import sqlite3
SQL_Data_connection = sqlite3.connect ('../input/pitchfork-data/database.sqlite')
artists = pd.read_sql_query ('SELECT * FROM artists', SQL_Data_connection)
check_q7 (artists)