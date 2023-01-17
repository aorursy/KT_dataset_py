import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame({'Apples':[30],'Bananas':[21]})
#check_q1(pd.DataFrame({'Apples':[30],'Bananas':[21]}))
#print(answer_q1())
d1 =pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
d1
#check_q2(d1)
#answer_q2()
d3 = pd.Series(data=['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name='Dinner')
d3
#check_q3(d3)
d4 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)
d4
#check_q4(d4)
d5 = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name='Pregnant Women Participating')
d5
check_q5(d5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
#d6 =pd.DataFrame.q6_df
df6 =q6_df.to_csv("cows_and_goats.csv")
check_q6(df6)

import sqlite3
conn= sqlite3.connect("../input/pitchfork-data/database.sqlite")
df7 = pd.read_sql_query("select * from artists",conn)
df7.head()
#check_q7(df7)