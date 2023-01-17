import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
check_q1(pd.DataFrame({'Apples': 30, 'Bananas': 21}, index=[0]))
check_q2(pd.DataFrame({"Apples": {"2017 Sales": 35, "2018 Sales": 41}, "Bananas": {"2017 Sales": 21, "2018 Sales": 34}}))

check_q3(pd.Series({"Flour": "4 cups", "Milk": "1 cup", "Eggs": "2 large", "Spam": "1 can"}, name="Dinner"))

check_q4(pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0))
check_q5(pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating'))

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv("cows_and_goats.csv"))
import sqlite3
check_q7(pd.read_sql(con=sqlite3.connect("../input/pitchfork-data/database.sqlite"), sql="SELECT * FROM artists;"))
