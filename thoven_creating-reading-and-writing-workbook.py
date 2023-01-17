import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
q1_df = pd.DataFrame({"Apples": [30], "Bananas": [21]})
check_q1(q1_df)
q2_df = pd.DataFrame({"Apples": [35, 41], "Bananas": [21, 34]}, index=["2017 Sales", "2018 Sales"])
check_q2(q2_df)
q3_ss = pd.Series(["4 cups", "1 cup", "2 large", "1 can"], name="Dinner", index=["Flour", "Milk", "Eggs", "Spam"])
check_q3(q3_ss)
q4_df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col="Unnamed: 0")
check_q4(q4_df)
q5_df = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name="Pregnant Women Participating")
check_q5(q5_df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
import sqlite3 as sql
conn = sql.connect("../input/pitchfork-data/database.sqlite")
c = conn.cursor()
c.execute("SELECT * FROM artists")
artists = c.fetchall()
reviewid = [artists[i][0] for i in range(len(artists))]
artist = [artists[i][1] for i in range(len(artists))]
reviewid
q7_df = pd.DataFrame({"reviewid": reviewid, "artist": artist})
check_q7(q7_df)