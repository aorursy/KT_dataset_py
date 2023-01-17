import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data = {'Apples':[30], 'Bananas':[21]}
df = pd.DataFrame(data=data)
print (df)
data = {'Apples':[35,21], 'Bananas':[41,34]}
df_ = pd.DataFrame(data=data, index=['2017 Sales', '2018 Sales'])
print (df_)
data = ['4 cups', '1 cup', '2 large', '1 can']
index = ['Flour', 'Milk', 'Eggs', 'Spam']
Dinner = pd.Series(data, index)
print (Dinner)
dataFrame = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
xlsData = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
cows_and_goats = q6_df.to_csv('cows_and_goats.csv')
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
artist = pd.read_sql_query('select * from artists',con=conn)
artist
