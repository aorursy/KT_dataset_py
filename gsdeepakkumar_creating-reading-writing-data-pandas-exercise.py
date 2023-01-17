import pandas as pd
pd.set_option('max_rows', 5)\


import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({'Apples':[30],'Bananas':21},index=[0])

check_q1(pd.DataFrame({'Apples':[30],'Bananas':21},index=[0]))
# Your code here
pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
#Check
check_q2(pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales']))
# Your code here
pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name="Dinner")
#Check
check_q3(pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name="Dinner"))
# Your code here 
wine_review=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)
#check
check_q4(pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0))
# Your code here
preg=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating")
check_q5(pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating"))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')

check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
import sqlite3
conn=sqlite3.connect('../input/pitchfork-data/database.sqlite')
#View table names in SQLite for querying:
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
review=pd.read_sql_query("Select * from artists",conn)
review.head()
#check answer
check_q7(pd.read_sql_query("Select * from artists",conn))