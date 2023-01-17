import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df_q1 = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
print(check_q1(df_q1))
df_q1
# Your code here
df_q2 = pd.DataFrame({'Apples': [35,41], 'Bananas': [21,34]}, index = ['2017 Sales', '2018 Sales'])
print(check_q2(df_q2))
df_q2
# Your code here
data = ['4 cups', '1 cup', '2 large', '1 can']
s_q3 = pd.Series(data, index= ['Flour', 'Milk', 'Eggs', 'Spam'], name = 'Dinner')
print(check_q3(s_q3))
s_q3
# Your code here 
df_q4 = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', usecols = range(1,11))
print(check_q4(df_q4))
df_q4
# Your code here
df_q5 = pd.read_excel(io = '../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
print(check_q5(df_q5))
df_q5.head(5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
d = q6_df.to_csv('cows_and_goats.csv')
check_q6(d)
# Your Code Here
import sqlite3

database = "../input/pitchfork-data/database.sqlite"
conn = sqlite3.connect(database)

q7_df = pd.read_sql_query("SELECT * FROM artists", conn)

conn.close()

print(check_q7(q7_df))
q7_df