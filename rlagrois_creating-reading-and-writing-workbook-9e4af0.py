import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
import sqlite3
check_q1(pd.DataFrame())
# Your code here
data1 = {"Apples":[30], "Bananas":[21]}
df1 = pd.DataFrame(data=data1)
check_q1(df1)
# Your code here
data2 = {"Apples":[35,41], "Bananas":[21,34]}
rows = [str((2017 + i)) + " Sales" for i in range(2)]

df2 = pd.DataFrame(data2, index = rows)

check_q2(df2)
# Your code here
data3 = {"Flour": "4 cups", "Milk":"1 cup", "Eggs": "2 large", "Spam": "1 can"}
ind = [i for i in data3]
series1 = pd.Series(data3, index= ind, name="Dinner", dtype="object")

series1
check_q3(series1)
# Your code here 
csv1 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
#csv1.head()
check_q4(csv1)
# Your code here
xls1 = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",
                    sheet_name="Pregnant Women Participating")

#xls1.head()
check_q5(xls1)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here

save = q6_df.to_csv("cows_and_goats.csv")
check_q6(save)
# Your Code Here
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
#c = conn.cursor()
query = "SELECT reviewid, artist FROM artists"
SQLdf = pd.read_sql(query, conn)

#SQLdf.head()
check_q7(SQLdf)

