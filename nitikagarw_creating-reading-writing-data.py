import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
apples = pd.Series([30])
bananas = pd.Series([21])
pd.DataFrame({'Apples' : apples, 'Bananas' : bananas})
# Your code here
index = ['2017 Sales', '2018 Sales']
df_2 = pd.DataFrame({'Apples': [35, 41], 'Bananas' : [21, 34]}, index = index)

check_q2(df_2)
# Your code here
pd.Series(['4 cups', '1 cup', '2 large', '1 can'], 
index=['Flour', 'Milk', 'Eggs', 'Spam'], 
name='Dinner')


# Your code here 
wine_review_dataframe = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
# wine_review_dataframe.head()
# wine_review_dataframe.describe()

# Your code here
women_dataframe = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name='Pregnant Women Participating')
# women_dataframe.columns
# women_dataframe.head()
check_q5(women_dataframe)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
check_q6(q6_df.to_csv('cows_and_goats.csv', sep=','))

# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
pd.read_sql_query("SELECT * FROM artists", conn)