import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
q1 = pd.DataFrame([{"Apples": 30, "Bananas": 21}])
check_q1(q1)
q2 = pd.DataFrame({"2017 Sales": {"Apples": 35, "Bananas": 21}, 
                    "2018 Sales": {"Apples": 41, "Bananas": 34}}).T
check_q2(q2)
q3 = pd.Series({"Flour": "4 cups", 
                "Milk": "1 cup", 
                "Eggs": "2 large", 
                "Spam": "1 can"}, 
               index={"Flour":0, 
                      "Milk": 1, 
                      "Eggs": 2, 
                      "Spam": 3}, 
               name="Dinner")
check_q3(q3)
q4 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
check_q4(q4)
q5 = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", 
                   sheet_name="Pregnant Women Participating")
check_q5(q5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv("cows_and_goats.csv"))
import sqlite3

connection = sqlite3.connect("../input/pitchfork-data/database.sqlite")

q7 = pd.read_sql("SELECT * FROM artists", connection)
check_q7(q7)
