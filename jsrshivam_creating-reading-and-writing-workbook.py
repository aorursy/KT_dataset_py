import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame([[30,21]], columns=['Apples', 'Bananas'])
check_q1(pd.DataFrame([[30,21]], columns=['Apples', 'Bananas']))
# Your code here
val1 = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'], index=['2017 Sales', '2018 Sales'])
print(val1)
check_q2(val1)
# Your code here
val3 = pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'})
val3.name = 'Dinner'
val3
print(check_q3(val3))

df4 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
print(check_q4(df4))
# Your code here
check_q5(pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name='Pregnant Women Participating'))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
q6_df.to_csv('cows_and_goats.csv')
check_q6()
import sqlite3

conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
df = pd.read_sql_query("select * from artists;", conn)# Your Code Here
df
check_q7(df)