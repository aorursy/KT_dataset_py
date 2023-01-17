import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
df=pd.DataFrame([[30,21]],columns=['Apples','Bananas'])
df
df=pd.DataFrame([[35,21],[41,34]],index=['2017 Sales','2018 Sales'],columns=['Apples','Bananas'])
df
d={'Flour': '4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}
s=pd.Series(d)
s
df=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
df.head()
dfx=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',index_col=0)
dfx
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cow_and_goats1.csv')
pwd
pd.read_csv('cow_and_goats1.csv',index_col=0)

import sqlite3
con=sqlite3.connect('../input/pitchfork-data/database.sqlite')
df=pd.read_sql_query('Select * from artists',con)
df
con=sqlite3.connect('../input/pitchfork-data/database.sqlite')
res = con.execute("SELECT name FROM sqlite_master WHERE type='table';") 
cnt=0
for name in res: 
    print(name[0])
    cnt +=1
print(cnt)