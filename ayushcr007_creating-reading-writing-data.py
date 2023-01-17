import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame()
from pandas import DataFrame
check_q1(DataFrame({'Apples':[30],'Bananas':[21]}))
# Your code here
from pandas import DataFrame
check_q2(DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales']))
# Your code here
from pandas import Series
Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
# Your code here 
import pandas as pd
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
check_q4(df.iloc[:,1:])
# Your code here
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
import sqlite3 as sql
df = sql.connect('../input/pitchfork-data/database.sqlite')
check_q7(pd.read_sql_query('select * from artists', df))