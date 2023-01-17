import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
d = {'Apples': [30],
        'Bananas':[21]}
df=pd.DataFrame(d)
d = {'Apples': pd.Series([35,41], index = ['2017 Sales', '2018 Sales'] ),
        'Bananas': pd.Series([21,34], index = ['2017 Sales', '2018 Sales'] )}
df =pd.DataFrame(d)
Dinner = pd.Series(data = ['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour', 'Milk', 'Eggs', 'Spam'])
wine_reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', sep= ',')
wic = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
import sqlite3

con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query('SELECT * from artists', con)
df.head()