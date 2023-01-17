import pandas as pd

pd.set_option('max_rows', 5)

from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data = {"Apples":[30], "Bananas":[21]}

df = pd.DataFrame(data, columns=["Apples","Bananas"])

print(df)

check_q1(df)
data = {"Apples":[35,41], "Bananas":[21,34], "Index":["2017 Sales","2018 Sales"]}

df = pd.DataFrame(data, columns=["Apples","Bananas"], index=data["Index"])

print(df)

check_q2(df)
Dinner = pd.Series(["4 cups", "1 cup", "2 large", "1 can"], index=["Flour","Milk","Eggs","Spam"], name="Dinner")

print(Dinner)

check_q3(Dinner)
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)

print(df)

check_q4(df)
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")

print(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
csv_file = q6_df.to_csv("cows_and_goats.csv")

check_q6(csv_file)
import sqlite3

conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")

artists = pd.read_sql_query("select * from artists", conn)

print(artists)

check_q7(artists)