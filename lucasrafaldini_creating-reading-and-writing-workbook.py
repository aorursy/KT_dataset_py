import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
df = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1(df)
df = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])
check_q2(df)
series = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
check_q3(series)
csv = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)
check_q4(csv)
xls = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
check_q5(xls)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

csv = q6_df.to_csv("cows_and_goats.csv")
check_q6(csv)
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')

artists = pd.read_sql_query("SELECT * FROM artists", conn)

print(str(check_q7(artists)) + '\n')
print(artists.head())