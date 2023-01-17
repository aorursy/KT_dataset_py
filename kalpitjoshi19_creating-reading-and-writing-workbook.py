import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df = pd.DataFrame({'Apples':[30],'Bananas':[21]})
df
# Your code here
data_dict = {'Apples':[35,41],'Bananas':[21,34]}
index_list = ['2017 Sales','2018 Sales']
df = pd.DataFrame(data_dict, index=index_list)
df
# Your code here
data_dict = {'Flour':'4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}
Dinner = pd.Series(data_dict, name='Dinner')
Dinner
# Your code here 
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',header=0)
df
# Your code here
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
df
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3
con = sqlite3.connect("../input/pitchfork-data/database.sqlite")
df = pd.read_sql_query("SELECT * from artists", con)
df