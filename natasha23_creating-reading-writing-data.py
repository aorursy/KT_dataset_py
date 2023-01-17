import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = {'Apple':[30],'Bananas':[21]}
df = pd.DataFrame(data,columns=['Apple','Bananas'])
print(df)
#check_q1(pd.DataFrame(data,columns=['Apple','Bananas']))
# Your code here
data = {'Apple':[35,41],'Bananas':[21,34]}
df = pd.DataFrame(data,columns=['Apple','Bananas'],index=['2017 Sales','2018 Sales'])
print(df)
# Your code here
series = pd.Series(['4 cups','1 cup','2 large','1 can'], index =['Flour','Milk','Eggs','Spam'])
print(series)
# Your code here 
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
print(df.head())
# Your code here
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
print(df.head())
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
