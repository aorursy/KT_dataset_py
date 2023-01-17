import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
dct = {'Apples': [30], 'Bananas': [21]}
df = pd.DataFrame(dct)
df
# Your code here
dct1 = {'Apples': [35, 41], 'Bananas': [21, 34]}
df1 = pd.DataFrame(dct1, index=['2017 Sales', '2018 Sales'])
df1
# Your code here
indices = ['Flour', 'Milk', 'Eggs', 'Spam']
data = ['4 cups', '1 cup', '2 large', '1 can']
ser = pd.Series(data, indices, name='Dinner')
ser
# Your code here 
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
df
# Your code here
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name=None)['Pregnant Women Participating']
df
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query('SELECT * FROM artists', conn)
df
