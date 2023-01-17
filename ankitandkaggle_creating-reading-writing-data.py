import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame({'Apples':[30],'Bananas':[21]})
# this can be created both ways, either using dict of dict or dict and index
print(pd.DataFrame({'Apples':{'2017 Sales':35,'2018 Sales':21},'Bananas':{'2017 Sales':41,'2018 Sales':34}}))
pd.DataFrame({'Guavas':[35,41],'Mangoes':[21,34]}, index = ['2017 Sales','2018 Sales'])
pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour','Milk','Eggs','Spam'], name = 'Dinner')
readwine = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)
readwine.head()
dataFrame = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name= 'Pregnant Women Participating')
dataFrame
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df

q6_df.to_csv('data.csv')
check = pd.read_csv('data.csv', index_col = 0)
check.head()
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
data = pd.read_sql_query("SELECT * FROM artists", conn)
data.head()