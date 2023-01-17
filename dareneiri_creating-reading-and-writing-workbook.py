import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
d = {'Apples': [30], 'Bananas': [21]}
df = pd.DataFrame(d)
check_q1(df)
d = {'Apples':[35, 41], 'Bananas':[21,34]}
df = pd.DataFrame(d).rename({0: '2017 Sales', 1: '2018 Sales'})
print(df)
check_q2(df)
series = pd.Series(data = ['4 cups', '1 cup', '2 large', '1 can'],\
                   index = ['Flour', 'Milk', 'Eggs', 'Spam'],\
                   name = 'Dinner')
print(series)
check_q3(series)
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
print(df)
check_q4(df)
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
print(df)
check_q5(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv('cows_and_goats.csv'))
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')

df = pd.read_sql_query("select * from artists;", conn)
conn.close()
print(df)
check_q7(df)