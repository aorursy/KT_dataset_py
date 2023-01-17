import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = pd.DataFrame({'Apples':[30],'Bananas':[21]})
check_q1(data)
# Your code here
check_q2(pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales']))
# Your code here
check_q3(pd.Series(['4 cups','1 cup','2 large','1 can'],index = ['Flour','Milk','Eggs','Spam'], name = 'Dinner'))
data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)
check_q4(data)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
check_q6(q6_df.to_csv("cows_and_goats.csv"))
# Your Code Here
import sqlite3 as sq
conn = sq.connect("../input/pitchfork-data/database.sqlite")
fires = pd.read_sql_query("SELECT * FROM artists",conn)
check_q7(fires)

