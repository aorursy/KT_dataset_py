import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
d = {'Apples' : pd.Series([30], index=[0]),
     'Bananas' : pd.Series([21], index=[0])}
df = pd.DataFrame(d)
check_q1(df)

d = {'Apples' : pd.Series([35, 41], index=['2017 Sales', '2018 Sales']),
     'Bananas' : pd.Series([21, 34], index=['2017 Sales', '2018 Sales'])}
df = pd.DataFrame(d)
check_q2(df)

s = pd.Series(['4 cups', '1 cup', '2 large', '1 can'],index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
check_q3(s)
winemag_data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
check_q4(winemag_data)

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
check_q6()
# Your Code Here