import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
a = pd.DataFrame({'Apples': [30], 'Bananas':[21]})
a
check_q1(a)
b = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index=['2017 Sales', '2018 Sales'])
b
check_q2(b)
# Your code here
c = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
c
check_q3(c)
# Your code here
file = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
file.head()
check_q4(file)
# Your code here
file2 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
file2
check_q5(file2)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
check_q6()
# Your Code Here