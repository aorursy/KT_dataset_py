import pandas as pd
%config IPCompleter.greedy=True
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
df=pd.DataFrame({'Apples':30,'Bananas':21},index=[0])
df
df=pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
df
df=pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name='Dinner')
df
df=pd.read_csv("..//input//wine-reviews//winemag-data_first150k.csv")
df
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
import sqlite3
con=sqlite3.Connection("..//input//pitchfork-data//database.sqlite")
df=pd.read_sql_query("SELECT * FROM artists",con)
df