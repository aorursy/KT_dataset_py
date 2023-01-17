import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame({'Apples' : [30], 'Bananas' : [20]})
pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]}, index =['2017 Sales','2018 Sales'])
pd.Series (['4 cups','1 cup','2 large','1 can'],
             index= ['Flour','Milk','Egg','Spam'],
             name = 'Dinner')
# Your code here 
wineReviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col =0)
wineReviews.head()
wineReviews.describe()
# Your code here
xlx = pd.ExcelFile ("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", index = 0)

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv ('cows_and_goats.csv')
# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
pd.read_sql_query('select * from artists',conn)