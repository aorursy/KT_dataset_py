import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
pd.DataFrame({'Apples': [30],'Bananas': [21]})
pd.DataFrame({'Apples': [35,41], 'Bananas': [21,34]}, index = ['2017 Sales','2018 Sales'])
pd.Series(['4 cups', '1 cup', '2 large', '1 can'], 
index=['Flour', 'Milk', 'Eggs', 'Spam'], 
name='Dinner')
#pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large','Spam':'1 can'}, name = "Dinner")
DataF = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
print(DataF.head())
DataF = pd.ExcelFile('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
print(DataF)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv',sep = '\t')
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
pd.read_sql_query("SELECT * FROM artists", conn)