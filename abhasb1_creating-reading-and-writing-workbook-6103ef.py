import pandas as pd
import numpy as np
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
arr = np.array([[30,21]])

check_q1(pd.DataFrame(arr,columns=['Apples','Bananas'],index=[0]))
arr = np.array([[35,21],[41,34]])
check_q2(pd.DataFrame(arr,columns=['Apples','Bananas'],index=['2017 Sales','2018 Sales']))
check_q3(pd.Series({'Flour':'4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}))
df = (pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv'))
check_q4(df.drop('Unnamed: 0',axis=1))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv("cows_and_goats.csv"))
import sqlite3
filepath = "../input/pitchfork-data/database.sqlite"
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
cur = conn.cursor()
cur.execute("select * from artists;")
results = cur.fetchall()
df2 = pd.DataFrame(results,columns=['reviewid','artist'])

check_q7(df2)

