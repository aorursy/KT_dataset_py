import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(df)
import pandas as pd
df = pd.DataFrame({"Apples":[30],
                   "Bananas":[21]})
print(df)
check_q1(df)
import pandas as pd
dfd = pd.DataFrame({"Apples":[35,41],
                   "Bananas":[21,34]},["2017 Sales","2018 Sales"])
print(dfd)
check_q2(dfd)
quantity = ['4 cups','1 cup','2 large','1 can']
items = ['Flour','Milk','Eggs','Spam']
ser1 = pd.Series(quantity,items)
print(ser1)
check_q3(ser1)
names = ['country','description','designation','points','price','province','region_1','region_2','variety','winery']
df7 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)
print(df7)
check_q4(df7)
dff = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
print(dff)
check_q5(dff)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
check_q6(q6_df)
import sqlite3
import numpy as np
connection = sqlite3.connect('../input/pitchfork-data/database.sqlite')
cursor = connection.cursor()
cursor.execute('select reviewid,artist from artists')
rows = cursor.fetchall()
indexes = np.arange(0,18831)
sqlDF = pd.DataFrame(rows,indexes,['reviewid','artist'])
print(sqlDF)
check_q7(sqlDF)