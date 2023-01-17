import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
ex1 = pd.DataFrame({"Apples": [30], "Bananas": [21]})
print(ex1)
print(check_q1(ex1))
# Your code here
ex2 = pd.DataFrame({"Apples": [35,41], "Bananas": [21,34]}, index=["2017 Sales", "2018 Sales"])
print(ex2)
print(check_q2(ex2))
# Your code here
ex3 = pd.Series(["4 cups", "1 cup", "2 large", "1 can"], index=["Flour", "Milk", "Eggs", "Spam"], name='Dinner')
print(ex3)
print(check_q3(ex3))
# Your code here 
filepath = "../input/wine-reviews/winemag-data_first150k.csv"
df = pd.read_csv(filepath, index_col=0) # use csv column for the index 
print(df)
print(check_q4(df))
# Your code here
filepath = "../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls"
df = pd.read_excel(filepath, 
                   sheet_name='Pregnant Women Participating')
print(check_q5(df))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
ex6 = q6_df.to_csv("cows_and_goats.csv")
print(check_q6(ex6))
# Your Code Here
filepath = "../input/pitchfork-data/database.sqlite"

# connect sqlite
import sqlite3
conn = sqlite3.connect(filepath)
file = pd.read_sql_query("SELECT * FROM artists", conn)
print(file)
print(check_q7(file))