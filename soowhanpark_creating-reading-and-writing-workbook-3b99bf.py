import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
a=pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1(a)
b=pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index=['2017 Sales', '2018 Sales'])
check_q2(b)
c=pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
check_q3(c)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
check_q4(wine_reviews)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
f=q6_df.to_csv("cows_and_goats.csv")
check_q6(f)
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
artists = pd.read_sql_query("SELECT * FROM artists", conn)
check_q7(artists)