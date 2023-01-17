import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = {'Apples': [30], 'Bananas': [21]}
df1 = pd.DataFrame(data=data)
check_q1(df1)
# Your code here
data2 = {'Apples': [35, 41], 'Bananas': [21, 34]}
df2 = pd.DataFrame(data=data2)
df2 = df2.rename(index={0:'2017 Sales', 1:'2018 Sales'})
print(df2)
check_q2(df2)
# Your code here
check_q3(pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner'))

# Your code here 
df5 = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
df5 = df5.drop(columns=['Unnamed: 0'])
check_q4(df5)
# Your code here
df5 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
print(df5)
check_q5(df5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
df6 = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
df6.to_csv('cows_and_goats.csv')
check_q6()
# Your Code Here

