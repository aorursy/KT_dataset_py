import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = { 'Apples':[30], 'Bananas': [21]}
df = pd.DataFrame(data)
df
check_q1(df)
#pd.DataFrame({'Apples': [30], 'Bananas': [21]})
# Your code here
pd.DataFrame({'Apples':[35,41], 'Bananas': [21,34]}, index = ['2017 Sales','2018 Sales'])
check_q2(pd.DataFrame({'Apples':[35,41], 'Bananas': [21,34]}, index = ['2017 Sales','2018 Sales']))
# Your code here
s = pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'}, index = ['Flour','Milk','Eggs','Spam'],name= 'Dinner')
s
check_q3(s)
# Your code here 
a=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',header = 0,index_col=0)
a
check_q4(a)
a = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
a
check_q5(a)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.head().to_csv('cows_and_goats.csv')
check_q6()
# Your Code Here
#import sqlite3
print(answer_q7())