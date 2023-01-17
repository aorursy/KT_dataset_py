import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q2(pd.DataFrame())
df_= pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1(df_)
df_= pd.DataFrame(index={'2017 Sales','2018 Sales'},data={'Apples': [35,41], 'Bananas': [21,34]})
check_q2(df_)
s_ = pd.Series(name='Dinner',data=['4 cups','1 cup','2 large','1 can'], index=['Flour','Milk','Eggs','Spam'])
check_q3(s_)
d_=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
check_q4(d_)
excel_=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
check_q5(excel_)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
check_q6(q6_df.to_csv('cows_and_goats.csv'))
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
df_=pd.read_sql_query("select * from artists", conn)