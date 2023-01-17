import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
d = {'Apples': [30], 'Bananas': [21]}
df = pd.DataFrame(data=d)
check_q1(df)
df = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])
check_q2(df)
s = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], 
              index = ['Flour', 'Milk', 'Eggs', 'Spam'],
              name = 'Dinner')
check_q3(s)
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)
check_q4(df)
tb = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
check_q5(tb)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv('cows_and_goats.csv'))
import sqlite3

con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
artists_dat = pd.read_sql_query("SELECT * from artists", con)

check_q7(artists_dat)