import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
ans = pd.DataFrame({'Apples':[30],'Bananas': [21]})
check_q1(ans)
# Your code here
ans = pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]}, index=['2017 Sales','2018 Sales'])
check_q2(ans)
# Your code here
ans = pd.Series(['4 cups','1 cup', '2 large', '1 can'], index=['Flour','Milk','Eggs','Spam'],name='Dinner')
check_q3(ans)
# Your code here 
ans = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
check_q4(ans)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
check_q6()
# Your Code Here
import sqlite3
con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
ans = pd.read_sql('select reviewid, artist from artists', con)
con.close()
check_q7(ans)
