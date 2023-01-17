import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Here we create the above df
dic = {'Apples': [30], 'Bananas': [21]}
fruits = pd.DataFrame(dic)
check_q1(fruits)
#print(answer_q1())
# Your code here
dic = {'Apples': [30,41], 'Bananas': [21,34]}
fruits = pd.DataFrame(dic,index=['2017 Sales','2018 Sales'])
check_q2(fruits)
print(fruits)
# Your code here
series= pd.Series(['4 cups','1 cup','2 large','1 can'], index = ['Flour', 'Milk', 'Eggs', 'Spam'])
check_q3(series)
# Your code here 
df_wine = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)
check_q4(df_wine)
#print(answer_q4())
# Your code here
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheetname='Pregnant Women Participating')
check_q5(df)
#print(answer_q5())
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here

check_q6(q6_df.to_csv("cows_and_goats.csv"))

# Your Code Here
import sqlite3
connection = sqlite3.connect("../input/pitchfork-data/database.sqlite")
pd.read_sql_query("SELECT * FROM artists", connection)
#check_q7(pd.read_sql_query("SELECT * FROM artists", connection))
print(pd.read_sql_query("SELECT * FROM artists", connection))