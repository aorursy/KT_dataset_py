import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
check_q1(pd.DataFrame({'Apples': [30], 'Bananas': [21]}))
#print(answer_q1())
check_q2(pd.DataFrame({'Apples': [35,41], 'Bananas': [21,34]}, index=['2017 Sales', '2018 Sales']))
check_q3(pd.Series(['4 cups','1 cup','2 large','1 can'], index=['Flour','Milk','Eggs','Spam']))
check_q4(pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0))
#print(answer_q4())
check_q5(pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating'))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv('cows_and_goats.csv'))
#check_q7(import sqlite3
#con = sqlite3.connect("../input/pitchfork-data/database.sqlite")
#cur = con.cursor()
# The result of a "cursor.execute" can be iterated over by row
#for row in cur.execute('SELECT * FROM artists;'):
#    print(row)
# Be sure to close the connection
#con.close()
#        )
print(answer_q7())