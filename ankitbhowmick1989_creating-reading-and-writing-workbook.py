import pandas as pd

pd.set_option('max_rows', 5)

from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())

print(answer_q1())
ans1 = pd.DataFrame({'Apples':[30],'Bananas': [21]})# Your code here

print(ans1)
ans2 = pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index = ['2017 Sales','2018 Sales'])# Your code here

print(ans2)
ans3 = pd.Series(['4 cups','1 cup','2 large','1 can'],index = ['Flour','Milk','Eggs','Spam'],name = 'Dinner')# Your code here

print(ans3)

win = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col = 0)# Your code here 
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')

print(answer_q7())

import sqlite3# Your Code Here

database = sqlite3.connect('../input/pitchfork-data/database.sqlite')

pd.read_sql_query('SELECT * FROM artists',database)


