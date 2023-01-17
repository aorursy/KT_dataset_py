import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
d = {'Apples': [30], 'Bananas': [21]}
pd.DataFrame(data = d)
check_q1(pd.DataFrame(data = d))
d = {'Apples': [35,41], 'Bananas': [21,34]}
pd.DataFrame(data = d, index = ['2017 Sales', '2018 Sales'])
check_q2(pd.DataFrame(data = d, index = ['2017 Sales', '2018 Sales']))
print(pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour', 'Milk', 'Eggs', 'Spam']))
check_q3(pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour', 'Milk', 'Eggs', 'Spam']))
check_q4(pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col = 0))

data = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name= 'Pregnant Women Participating')
data.head()
check_q5(data)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.head()
check_q6(q6_df.to_csv("cows_and_goats.csv"))
# Your Code Here