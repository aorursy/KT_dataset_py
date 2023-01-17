import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
q1_data = {"Apples": [30], "Bananas": [21]}
q1_df = pd.DataFrame(q1_data, columns=["Apples", "Bananas"])
check_q1(q1_df)
# Your code here
q2_data = {"Apples": [35, 41], "Bananas": [21, 34]}
q2_df = pd.DataFrame(q2_data, index=["2017 Sales", "2018 Sales"] ,columns=["Apples", "Bananas"])
check_q2(q2_df)
# Your code here
q3_series = pd.Series(["4 cups", "1 cup", "2 large", "1 can"], index=["Flour", "Milk", "Eggs", "Spam"], name="Dinner", dtype=np.object)
check_q3(q3_series)
# Your code here 
data_wine = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", header=0, index_col=0)
check_q4(data_wine)
# Your code here
excel_data = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name="Pregnant Women Participating")
check_q5(excel_data)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
f = q6_df.to_csv('cows_and_goats.csv')
check_q6(f)
# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
table_names = conn.execute("SELECT * from sqlite_master where type='table';")
for t1 in table_names:
    print(t1[1])
df_7 = pd.read_sql_query("SELECT * from artists", conn)
check_q7(df_7)