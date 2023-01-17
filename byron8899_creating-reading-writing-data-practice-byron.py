import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
data = {'Apples':[30], 'Bananas':[21]}

df= pd.DataFrame(data, columns=['Apples', 'Bananas'])
check_q1(df)
data = {'Apples':[35,41], 'Bananas':[21,34]}
df= pd.DataFrame(data, columns=['Apples', 'Bananas'], index=['2017 Sales', '2018 Sales'])
check_q2(df)
data = ['4 cups', '1 cup', '2 large', '1 can']
s = pd.Series(data,['Flour','Milk','Eggs','Spam'])
s
check_q3(s)
data = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
df = data.drop(columns=['Unnamed: 0'])
check_q4(df)
data = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
check_q5(data)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
cows_and_goats = q6_df.to_csv('cows_and_goats.csv')
check_q6(cows_and_goats)
import sqlite3
# Create SQL connection to Kaggles SQLite Database (database.sqlite)
con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query('SELECT * FROM artists', con)
df
check_q7(df)