import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
df = pd.DataFrame({'Apples':[30],'Bananas':[21]},columns=['Apples','Bananas'])
print(check_q1(df))
df = pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
print(check_q2(df))
ser = pd.Series({'Flour': '4 cups','Milk':'1 cup','Eggs':'2 large', 'Spam': '1 can'}, name='Dinner')
print(check_q3(ser))
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
df = df.drop('Unnamed: 0',axis=1)
print(check_q4(df))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
print(check_q6(q6_df.to_csv("cows_and_goats.csv")))
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
d = conn.execute("SELECT * FROM artists")
dd = d.fetchall()
df=pd.DataFrame(dd,columns=['reviewid','artist'])
print(check_q7(df))