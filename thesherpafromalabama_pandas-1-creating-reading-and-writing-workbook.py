import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
df_ab=pd.DataFrame({'Apples': [30], 'Bananas': [21]})
#pd.DataFrame?
print(df_ab)
check_q1
# Your code here
df_ab=pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]},
                    index=['2017 Sales', '2018 Sales'])

check_q2
# Your code here
pd.Series(['4 cups', 1, 2,1], index=['Flour', 'Milk', 'Eggs','Spam'], name='Dinner')

#Note, index and passed values MUST MATCH
# Your code here 
df_wine=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',
                   index_col=0)
print(df_wine)
df_wine.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv') #Put quotes around new file name
check_q6
# Your Code Here
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
artists = pd.read_sql_query("SELECT * FROM artists", conn)
artists.head()
