import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
d = {'Apples': [30], 'Bananas': [21]}
pd.DataFrame(data = d)
# check the answer
check_q1(pd.DataFrame(data = {'Apples': [30], 'Bananas': [21]}))
d = {'Apples':[35,41],'Bananas':[21,34]}
df = pd.DataFrame(data=d, index = ['2017 Sales','2018 Sales'])
df
# check the answer
check_q2(pd.DataFrame(data = {'Apples':[35,41],'Bananas':[21,34]}, index = ['2017 Sales','2018 Sales']))
# Your code here
n3 = pd.Series(['4 cups','1 cup','2 large','1 can'], index = ['Flour','Milk','Eggs','Spam'],name = 'Dinner')
n3
# check the answer
check_q3(n3)
# Your code here 
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
check_q4(pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0))
# Your code here
pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name = 'Pregnant Women Participating' )
# check answer
check_q5(pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name = 'Pregnant Women Participating' ))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
q6_df.to_csv("cows_and_goats.csv")
print(check_q6())
# Your Code Here
import sqlite3 as sql
answer = pd.read_sql_query("SELECT * FROM artists",con = sql.connect('../input/pitchfork-data/database.sqlite'))
answer
check_q7(answer)
