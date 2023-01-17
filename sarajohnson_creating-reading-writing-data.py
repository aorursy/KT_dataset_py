import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
print(answer_q1())
#Create a dataframe; print it; check if it's correct
df1 = pd.DataFrame({'Apples':[30],'Bananas':[21]})
print(df1)
check_q1(df1)
# Create dataframe; print it; check it
df2 = pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
print(df2)
check_q2(df2)
#Create a series, print it, check it
series1 = pd.Series(['4 cups','1 cup','2 large','1 can'], index=['Flour','Milk','Eggs','Spam'],name='Dinner')
print(series1)
check_q3(series1)
#Read csv into a dataframe; print it; check it
df3 = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
print(df3.head())
check_q4(df3)
#Read xls into dataframe; print; check
df4 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
print(df4.head())
check_q5(df4)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
#Save existing dataframe to your HD; print; check
q6_df.head().to_csv('cows_and_goats.csv')
check_q6()
#Read SQL; print; check
import sqlite3 as sql
conn = sql.connect('../input/pitchfork-data/database.sqlite')
pitchfork = pd.read_sql_query('SELECT * FROM artists', conn)
print(pitchfork)
check_q7(pitchfork)