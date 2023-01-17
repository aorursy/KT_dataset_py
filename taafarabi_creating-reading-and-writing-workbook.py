import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
q1_df = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
q1_df
q2_df = pd.DataFrame({'Apples': [35,41], 'Bananas': [21,34]}, index=['2017 Sales','2018 Sales'])
q2_df
data = {"Flour":"4 cups", "Milk":"1 cup","Eggs":"2 large", "Spam":"1 can"}
q3_sr = pd.Series(data, index = ['Flour','Milk', 'Eggs','Spam'], name = 'Dinner')
q3_sr
q4 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
q4 = q4.iloc[:,1:]
q4


q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
check_q6()
import sqlite3
c = sqlite3.connect('../input/pitchfork-data/database.sqlite')

df = pd.read_sql_query("SELECT * FROM artists", c)
df
#check_q7(df)