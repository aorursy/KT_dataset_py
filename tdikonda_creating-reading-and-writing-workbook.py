import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data = {'Apples':[30], 'Bananas':[21]} 
df = pd.DataFrame(data, columns=['Apples','Bananas'])
check_q1(df)
data = {'Apples':[35,41], 'Bananas':[21,34]} 
df = pd.DataFrame(data,index=['2017 Sales','2018 Sales'] , columns=['Apples','Bananas'])
check_q2(df)
s = pd.Series(data = ['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Diner')
check_q3(s)
pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
check_q4(df)
answer_q4()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
check_q6()
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
pd.read_sql_query('SELECT * from artists',conn)
#check_q7()
#answer_q7()