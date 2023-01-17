import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
d = {'Apples':[30], 'Bananas':[21]}
df = pd.DataFrame(data=d)
check_q1(df)
# Your code here
d2 = {'Apples':[35,41], 'Bananas':[21,34]}
i2 = ['2017 Sales', '2018 Sales']
df2 = pd.DataFrame(data=d2, index=i2)
check_q2(df2)
# Your code here
d3 = ['4 cups', '1 cup', '2 large', '1 can']
i3 = ['Flour', 'Milk', 'Eggs', 'Spam']
dser = pd.Series(data=d3, index=i3, name='Dinner')
check_q3(dser)
# Your code here 
d4 = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
check_q4(d4)
# Your code here
d5 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
check_q5(d5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
d6=q6_df.to_csv('cows_and_goats.csv')
check_q6(d6)
# Your Code Here
import sqlite3

con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
d7 = pd.read_sql_query("SELECT * from artists", con)
check_q7(d7)