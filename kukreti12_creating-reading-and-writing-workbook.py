import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
#First Excercise
d = {'Apples':[30],'Bananas':[21]}
df = pd.DataFrame(data=d)
df
check_q1(df)
df = pd.DataFrame({'set':['2017 Sales','2018 Sales'],'Apples':[35,41],'Bananas':[21,34]})

df.set_index('set', inplace=True)
df = df.rename_axis(None)
check_q2(df)
s = {'Flour':'4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}
s2= pd.Series(s,name='Dinner')
check_q3(s2)
s2
review = pd.DataFrame.from_csv("../input/wine-reviews/winemag-data_first150k.csv")
check_q4(review)
x= pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating")
y = pd.DataFrame(x)
check_q5(y)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
disc = q6_df.to_csv('cows_and_goats.csv')
disc
check_q6(disc)
import sqlite3
conn =sqlite3.connect('../input/pitchfork-data/database.sqlite')
xy = pd.read_sql("Select * from artists", conn)
check_q7(xy)