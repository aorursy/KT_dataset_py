import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame({'Apples': [30], 'Bananas': [21]})
pd.DataFrame({'Apples': [35,41], 'Bananas': [21,34]},index=['2017 Sales','2018 Sales'])
# Your code here
pd.Series(['4 cups','1 cup','2 large','1 can'], index=['Flour','Milk','Eggs','Spam'],name='Dinner')
# Your code here 
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col='Unnamed: 0')
pd.DataFrame(wine_reviews)
# Your code here
a = pd.read_csv("../input/publicassistance/WICAgencies2014ytd/Pregnant_Women_Participating.csv")
a.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv',index=False)
# Your Code Here
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query('select * from artists',conn)
df