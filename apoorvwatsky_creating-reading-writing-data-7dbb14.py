import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
dat = pd.DataFrame({'Apples' : [30], 'Bananas' : [21]})
check_q1(dat)
# Your code here
dat2 = pd.DataFrame({'Apples' : [35, 41], 'Bananas' : [21, 34]}, index = ['2017 Sales', '2018 Sales'])
check_q2(dat2)
# Your code here
dat_series = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour', 'Milk', 'Eggs', 'Spam'])
check_q3(dat_series)
# Your code here 
wine = pd.DataFrame.from_csv('../input/wine-reviews/winemag-data_first150k.csv')
check_q4(wine)
# Your code here
df_excel = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
check_q5(df_excel)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
import sqlite3 as sq
art = sq.connect('../input/pitchfork-data/database.sqlite')
art_attack = pd.read_sql_query('select * from artists', art)
check_q7(art_attack)