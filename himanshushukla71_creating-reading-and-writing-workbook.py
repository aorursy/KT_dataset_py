import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
d=[[30,21]]
df=pd.DataFrame(d,columns=['Apple','Banana'])
print(df)
data={'Apples':[35,41],'Banana':[21,34]}
df=pd.DataFrame(data,index=['2017 Sales','2018 Sales'])
df
import numpy as np
data=np.array(['4 cups','1 cup','2 large','1 can'])
pd.Series(data,index=['Flour','Milk','Eggs','Spam'],name='Dinner')

wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv",index_col=0)
wine_reviews.head()
wic=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2013ytd.xls", 
                    sheet_name='Pregnant Women Participating')
wic
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
artists=pd.read_sql_query("select * from artists",conn)
artists