import pandas as pd
maxrows = 5 #5
pd.set_option('max_rows', maxrows)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
a1 = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1(a1)
# Your code here
a2 = pd.DataFrame({'Apples': [35,41], 'Bananas': [21,34]}, index = ['2017 Sales', '2018 Sales'])
check_q2(a2)
# Your code here
a3 = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour', 'Milk','Eggs','Spam'], name = 'Dinner')
check_q3(a3)
# Your code here 
a4 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col = 0)
check_q4(a4) 

a4
# Your code here
a5 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
check_q5(a5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
a6 = q6_df.to_csv('cows_and_goats.csv')
check_q6()
check_q6
# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
artists = pd.read_sql_query("SELECT * FROM artists", conn)
check_q7(artists)