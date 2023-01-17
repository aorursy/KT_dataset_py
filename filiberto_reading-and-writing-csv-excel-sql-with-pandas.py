import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame({'Bananas':[21], 'Apples': [30]})
pd.DataFrame({'Bananas':[21,34], 'Apples': [35,41]},
            index=['2017 Sales', '2018 Sales'])
pd.Series(['cups','cup','large','can'], index=['Flour 4', 'Milk 1', 'Eggs 2','Spam 1'], name='Dinner')
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
wic = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2013ytd.xls", sheet_name='Pregnant Women Participating')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")

artists = pd.read_sql_query("SELECT * FROM artists", conn)
artists.head()