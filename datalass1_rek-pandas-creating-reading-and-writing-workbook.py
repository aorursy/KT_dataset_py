import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q2(pd.DataFrame())
# Your code here
pd.DataFrame({'Apples':[30], 'Bananas':[21]})
#check the answer
check_q1(pd.DataFrame({'Apples':[30], 'Bananas':[21]}))
# Your code here
pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]}, index=['2017 Sales', '2018 Sales'])
# Your code here
pd.Series({'Flour':'4 cups', 'Milk':'1 cup', 'Eggs': '2 large', 'Spam':'1 can'}, name='Dinner')
# Your code here 
path = '../input/wine-reviews/winemag-data_first150k.csv'
wine_df = pd.read_csv(path)
wine_df.head(2)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3
#connect to the SQL db
conn = sqlite3.Connection('../input/pitchfork-data/database.sqlite')
#select the data 
artists = pd.read_sql_query('SELECT * FROM artists', conn)
artists
