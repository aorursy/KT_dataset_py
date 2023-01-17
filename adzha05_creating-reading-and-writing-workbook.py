import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
fruits = ['Apples', 'Bananas']
values = [(30, 21)]
df = pd.DataFrame(values, columns = fruits)
check_q1(pd.DataFrame(df))
print(answer_q1())
df = pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]}, index=['2017 Sales', '2018 Sales'])
s = pd.Series(['4 cups','1 cup','2 large','1 can'], index = ['Flour','Milk','Eggs', 'Spam'], name='Dinner')
s
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', header=None, nrows=5) 

pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
import sqlite3
conn = sqlite3.conncet("../input/pitchfork-data/database.sqlite")
pd.read_sql_query("SELECT * FROM artists", conn)