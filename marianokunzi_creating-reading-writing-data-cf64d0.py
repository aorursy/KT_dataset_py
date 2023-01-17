import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
d = {'Apples': [30], 'Bananas': [21]}
df = pd.DataFrame(d)
check_q1(df)
d = {'Apples': [35, 41], 'Bananas': [21, 34]}
df = pd.DataFrame(d, index=['2017 Sales', '2018 Sales'])
print(df)
check_q2(df)
s = pd.Series(
        ['4 cups', '1 cup', '2 large', '1 can'],
        index=['Flour', 'Milk', 'Eggs', 'Spam'],
        name='Dinner')
print(s)
check_q3(s)
a = pd.read_csv(
        '../input/wine-reviews/winemag-data_first150k.csv', 
        index_col=0)
print(a)
check_q4(a)
b = pd.read_excel(
    '../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',
    sheet_name='Pregnant Women Participating')

print(b)
check_q5(b)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv('cows_and_goats.csv'))
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query("SELECT * FROM artists", conn)
print(df.head())
check_q7(df)