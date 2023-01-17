import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here

df = pd.DataFrame({'Appples':[30], 'Bananas':21}, index=[0])

df


# Your code here

df1 = pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]}, index = ['2017 Sales', '2018 Sales'])

df1

# Your code here

df2 = pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs':'2 large', 'Spam' : '1 can'}, name='Dinner', index=['Flour', 'Milk', 'Eggs', 'Spam'])

df2


# Your code here 

df4 = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', usecols=['country','description','designation','points','price', 'province', 'region_1', 'region_2', 'variety', 'winery'])

df4


# Your code here

df5 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')

df5

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here

q6_df.to_csv('cows_and_goats.csv')

# Your Code Here

import sqlite3

con = sqlite3.connect('../input/pitchfork-data/database.sqlite')

df_op = pd.read_sql(sql='select reviewid, artist from artists', con=con )

df_op
