import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({'Apples': [30], 'Bananas': [21]})
# Your code here
fruits={}
dataf = pd.DataFrame(data={'Apple':[35,41],'Bananas':[21,34]}, index=['2017 Sales', '2018 Sales'])
print(dataf)
# Your code here
Din = pd.Series(data={'Flour': '4 cups','Milk': '1 cup','Eggs':'2 large','Spam':'1 can'},name='Dinner')
Din
# Your code here 
data = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
# Your code here
xldataf = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
writedf = q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3
connection = sqlite3.connect('../input/pitchfork-data/database.sqlite')
pd.read_sql_query('SELECT * FROM artists', connection)
#print(answer_q7())
