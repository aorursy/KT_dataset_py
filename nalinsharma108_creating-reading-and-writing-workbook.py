import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({'Apples': [30], 'Bananas': 21})
print(check_q1(pd.DataFrame({'Apples': [30], 'Bananas': 21})))
print(answer_q1())
# Your code here
pd.DataFrame({'Apples': [35,41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])
print(check_q2(pd.DataFrame({'Apples': [35,41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])))
print(answer_q2())
# Your code here
#pd.Series({'Flour': '4cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'},  name = 'Dinner')
pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
print(check_q3(pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')))
print(answer_q3())
# Your code here 
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col= 0)
print( check_q4(pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)))
print( answer_q4()) 
# Your code here
pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
print( check_q5(pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')))
print( answer_q5()) 
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_goats.csv')
print (answer_q6())
# Your Code Here
import sqlite3
con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query('select reviewid, artist from artists', con)
df.head()
print(answer_q7())
