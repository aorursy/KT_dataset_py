import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
q1 = pd.DataFrame({'Apples':30, 'Bananas':21}, index=[0])
check_q1(q1)
#print(answer_q1())
# Your code here
q2 = pd.DataFrame({'Apples': [35, 41],
             'Bananas': [21, 34]},
            index=['2017 Sales', '2018 Sales'])
check_q2(q2)
# Your code here
q3 = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
check_q3(q3)
# Your code here 
q4_winereviews =pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
q4_winereviews.head()
check_q4(q4_winereviews)
# Your code here
q5 = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name="Pregnant Women Participating")
q5
check_q5(q5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
q7 = pd.read_sql_query("select * from artists", conn)
check_q7(q7)