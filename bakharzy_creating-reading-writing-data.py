import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = {'Apples':[30],'Bananas':[21]}
df = pd.DataFrame(data,columns=['Apples','Bananas'])
df
check_q1(df)


# Your code here
d = {'Apples':[35,41],'Bananas':[21,34]}
df = pd.DataFrame(d,index=['2017 Sales','2018 Sales'])
df
check_q2(df)
# Your code here
s = pd.Series(['4 cups','1 cup','2 large','1 can'],index = ['Flour','Milk','Eggs','Spam'], name = 'Dinner')
s
check_q3(s)
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
df
check_q4(df)
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
df
check_q5(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here