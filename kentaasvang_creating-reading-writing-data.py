import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Creating the dict with the values given
table_for_e1 = {"Apples": [30], "Bananas": [21]}

# Turning it into a DataFrame
df_for_e1 = pd.DataFrame(table_for_e1)

# Checking the exercise
print(df_for_e1)
print("\n")
print(check_q1(df_for_e1))
# create the dict
table_for_e2 = {"Apples": [35, 41], "Bananas": [21, 34]}

# Declare index
index_for_e2 = ["2017 Sales", "2018 Sales"]

# Create pandas.DataFrame()
df_for_e2 = pd.DataFrame(table_for_e2, index=index_for_e2)

# Checking the exercise
print(df_for_e2)
print("\n")
print(check_q2(df_for_e2))
# Create python-list of values
list_for_e3 = ["4 cups", "1 cup", "2 large", "1 can"]

# Create index
index_for_e3 = ["Flour", "Milk", "Eggs", "Spam"]

# Create pandas.Series()
series_for_e3 = pd.Series(list_for_e3, index=index_for_e3, name="Dinner")

# Check exercise
print(series_for_e3)
print("\n")
print(check_q3(series_for_e3))
# filepath to the csv-file
csv_path_e4 = "../input/wine-reviews/winemag-data_first150k.csv"

# load csv file into dataframe
df_for_e4 = pd.read_csv(csv_path_e4)

# make Unnamed-column the index and drop the column from the pandas.DataFrame()
df_for_e4.set_index(df_for_e4["Unnamed: 0"], inplace=True)
df_for_e4.drop(["Unnamed: 0"], axis="columns", inplace=True)

# Check the dataframe
print(df_for_e4.head())
print("\n")
print(check_q4(df_for_e4))
# read xls into pandas.DataFrame()
df_for_e5 = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheetname="Pregnant Women Participating")

# Check exercise
check_q5(df_for_e5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# write q6_df to disc
q6_df.to_csv("cows_and_goats.csv")

check_q6(q6_df)
import sqlite3  # libraru for connecting to SQLite3 database
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")  # connects to the database

# get data from database 
artists = pd.read_sql_query("SELECT * FROM artists", conn)

check_q7(artists)
