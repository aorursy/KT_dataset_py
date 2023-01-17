import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df  = pd.DataFrame({'Apples':[30], 'Bananas': [21]})
print(check_q1(df))
# Your code here
df2 = pd.DataFrame({'Apples': [35, 41], 'Bananas':[21, 34]}, index=['2017 Sales', '2018 Sales'])
check_q2(df2)
# Your code here
s = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'])
check_q3(s)
# Your code here 
df_wine = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
df_wine.head()
check_q4(df_wine)
# Your code here
df_excel = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating',)
df_excel.head()
check_q5(df_excel)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
check_q6('cows_and_goats.csv')
# Your Code Here
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
sql_table = pd.read_sql_query('SELECT * FROM artists', conn)
check_q7(sql_table)
