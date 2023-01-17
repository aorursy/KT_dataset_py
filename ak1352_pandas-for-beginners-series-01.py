import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({'Apple':['30'] , 'Bananas':['21']})
# Your code here
pd.DataFrame({'Apple':['30'] , 'Bananas':['21']} ,  index=['2017 Sales' , '2018 Sales'])
# Your code here
pd.Series(['4 cups' , '1 cup' , '2 large' , '1 can'] , index = ['Flour' , 'Milk' , 'Eggs' , 'Spam'] , name = 'Dinner')
# Your code here 
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')# reading csv file into a dataframe
df.head() # first five rows only
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3 # python library for sql
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite") # making connection to database
df_sql = pd.read_sql_query('Select * from artists' , conn) # run sql query to get data from database into a dataframe
df_sql.head() # first five rows