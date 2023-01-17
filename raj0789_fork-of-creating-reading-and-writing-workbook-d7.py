import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame({'Applies':[30], 'Bananas':[21]})
pd.DataFrame({'Applies':[35,41], 'Bananas':[21,34]},
            index=['2017 Sales','2018 Sales'])
pd.Series(['4 cups','1 cups','2 large','1 can'], index=['Flour','Milk','Eggs','Spam'], name='Dinner')
wine_review=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
wine_review
public_assistance=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",
                               sheet_name='Pregnant Women Participating')
public_assistance.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
import sqlite3 as sp
artists = sp.connect("../input/pitchfork-data/database.sqlite")
pitchfork = pd.read_sql_query("SELECT * from artists", artists)
pitchfork.head()