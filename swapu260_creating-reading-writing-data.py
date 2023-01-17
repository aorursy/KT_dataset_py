import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df = pd.DataFrame({
    'Apples':[30], 'Bananas':[21]
})
df
# Your code here
data = {'Apples':[35, 41], 'Bananas': [21, 34]}
df = pd.DataFrame(data, index=['2017 Sales', '2018 Sales'])
check_q2(df)
# Your code here
data = ['4 cups', '1 cup', '2 large', '1 can']
index = ['Flour', 'Milk', 'Eggs', 'Spam']
series = pd.Series(data=data, index= index, name='Dinner')
check_q3(series)
# Your code here 
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
df.head()
# Your code here
xls = pd.ExcelFile('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
df = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
check_q5(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here-
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
import sqlite3

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("../input/pitchfork-data/database.sqlite")
df = pd.read_sql_query("SELECT * from 'artists'", con)
check_q7(df)
