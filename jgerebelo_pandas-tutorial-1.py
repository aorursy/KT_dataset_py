import pandas as pd

pd.set_option('max_rows', 5)



import sys

sys.path.append('../input/advanced-pandas-exercises/')

from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here

ex_1 = {'Apples': [30], 'Bananas': [21]}

pd.DataFrame(ex_1)
check_q1(pd.DataFrame(ex_1))
# Your code here

ex_2 = {'Apples': [35, 41], 'Bananas': [21, 34]}

pd.DataFrame(ex_2, index = ['2017 Sales', '2018 Sales'])
check_q2(pd.DataFrame(ex_2, index = ['2017 Sales', '2018 Sales']))
# Your code here

ex_3 = ['4 cups', '1 cup', '2 large', '1 can']

pd.Series(ex_3, index = ['Flour', 'Milk', 'Eggs', 'Spam'], name = 'Dinner')
check_q3(pd.Series(ex_3, index = ['Flour', 'Milk', 'Eggs', 'Spam'], name = 'Dinner'))
# Your code here 

pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)
check_q4(pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0))
# Your code here

pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
check_q5(pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating'))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here

q6_df.to_csv('cows_and_goats.csv')
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here

import sqlite3



ex_7 = sqlite3.connect('../input/pitchfork-data/database.sqlite')



pd.read_sql_query('SELECT * FROM artists', ex_7)
check_q7(pd.read_sql_query('SELECT * FROM artists', ex_7))