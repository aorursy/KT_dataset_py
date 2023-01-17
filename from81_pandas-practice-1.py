import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df = pd.DataFrame({'Apples':[30], 'Bananas':[21]})
print(check_q1(df))
data = {'Apples':[35, 41], 'Bananas':[21, 34]}
df = pd.DataFrame(data, index=['2017 Sales','2018 Sales'])
print(check_q2(df))
# Your code here
s = pd.Series({'Flour':'4 cups', 'Milk':'1 cup', 'Eggs':'2 large', 'Spam':'1 can'})
print(check_q3(s))
# Your code here 
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
print(check_q4(df))
# Your code here
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
print(check_q5(df))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
print(check_q6(q6_df.to_csv('cows_and_goats.csv')))
# Your Code Here
import sqlite3
sql='SELECT reviewid, artist FROM artists'
con=sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query(sql, con)
print(check_q7(df))
