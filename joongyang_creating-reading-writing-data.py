import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
apples = [30]
bananas = [21]
fruits = {'Apples': apples, 'Bananas': bananas}
fruits = pd.DataFrame(fruits)

check_q1(fruits)
# Your code here
apples = [35, 41]
bananas = [21, 34]
years = ['2017 Sales', '2018 Sales']
fruits = {'Apples': apples, 'Bananas': bananas}
fruits = pd.DataFrame(fruits, index=years)

print(fruits)
check_q2(fruits)
# Your code here
values = ['4 cups', '1 cup', '2 large', '1 can']
items = ['Flour', 'Milk', 'Eggs', 'Spam']
inv = pd.Series(values, index=items)

check_q3(inv)
# Your code here 
wine = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

check_q4(wine)
# Your code here
df = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",
                   sheet_name="Pregnant Women Participating")

check_q5(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv("cows_and_goats.csv")

check_q6(q6_df)
# Your Code Here
import sqlite3
con = sqlite3.connect("../input/pitchfork-data/database.sqlite")
cur = con.cursor()
cur.execute("select reviewid, artist from artists;")
df = cur.fetchall()

colnames = ["reviewid", "artist"]
df = pd.DataFrame(df, columns=colnames)
df.head()

check_q7(df)