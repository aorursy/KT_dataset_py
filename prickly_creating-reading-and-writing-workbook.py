import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
dict = {"Apples":[30], "Bananas":[21]}
df = pd.DataFrame(dict)
check_q1(df)
# Your code here
dict = {"Apples":[35,41], "Bananas":[21,34]}
df = pd.DataFrame(dict, index=["2017 Sales", "2018 Sales"])
#print(df.head())
check_q2(df)
# Your code here
ser = pd.Series({"Flour": "4 cups", "Milk": "1 cup", "Eggs": "2 large", "Spam":"1 can"})
check_q3(ser)
# Your code here 
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
#print(df)
check_q4(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv("cows_and_goats.csv")
q6_df_1 = pd.read_csv("cows_and_goats.csv", index_col=0)
print(q6_df_1.head())
# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
reviews = pd.read_sql_query("SELECT reviewid, artist FROM artists", conn)
print(reviews)