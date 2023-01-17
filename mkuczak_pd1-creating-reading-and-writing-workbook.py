import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df1 = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1(df1)
# Your code here
df2 = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21,34]}, index=['2017 Sales', '2018 Sales'])
check_q2(df2)
# Your code here
s3 = pd.Series(['4 cups','1 cup','2 large','1 can'], index=['Flour','Milk','Eggs','Spam'])
check_q3(s3)
# Your code here
df4 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
check_q4(df4)
# Your code here
df5 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
check_q5(df5)
df6 = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
csv6 = df6.to_csv("cows_and_goats.csv")
check_q6(csv6)
# Your Code Here