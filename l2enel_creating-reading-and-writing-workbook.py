import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q7(q7)
q1 = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
q2 = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index=['2017 Sales', '2018 Sales'])
q2
q3 = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
q3
file_path = '../input/wine-reviews/winemag-data_first150k.csv'
df = pd.read_csv(file_path, index_col=0)
df.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')

q7 = pd.read_sql_query("SELECT * FROM artists", conn)
q7.head()