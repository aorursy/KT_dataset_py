import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = {"Apples":[30], "Bananas":[21]}
df = pd.DataFrame(data)
print(df)
check_q1(df)
# Your code here
data = {"Apples":[35,41], "Bananas":[21,34]}
df = pd.DataFrame(data, columns=["Apples","Bananas"], index=['2017 Sales','2018 Sales'])
print(df)
check_q2(df)
# Your code here
df= pd.Series(["4 cups", "1 cup", "2 large", "1 can"], index=["Flour","Milk","Eggs","Spam"], name="Dinner")
print(df)
check_q3(df)
# Your code here 
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)
print(df)
check_q4(df)
# Your code here
df = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name="Pregnant Women Participating")
print(df)
check_q5(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
print(q6_df)
csv_file = q6_df.to_csv("cows_and_goats.csv")
check_q6(csv_file)
# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
artists = pd.read_sql_query("select * from artists", conn)
print(artists)
check_q7(artists)