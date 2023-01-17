import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
dataQ1 = {'Apples': [30], 'Bananas': [21]}
dataFrameQ1 = pd.DataFrame(data=dataQ1)
check_q1(dataFrameQ1)
dataFrameQ2 = pd.DataFrame(
    data={'Apples': [35,41], 'Bananas': [21,34]}, 
    index=['2017 Sales', '2018 Sales']
)
check_q2(dataFrameQ2)
seriesQ3 = pd.Series(
    data=['4 cups', '1 cup', '2 large', '1 can'],
    index=['Flour', 'Milk', 'Eggs', 'Spam'],
    name='Dinner'
)
check_q3(seriesQ3)
dataFrameQ4 = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
check_q4(dataFrameQ4)
dataFrameQ5 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
check_q5(dataFrameQ5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
dataFrameQ7 = pd.read_sql_query("SELECT * FROM artists", conn)
check_q7(dataFrameQ7)