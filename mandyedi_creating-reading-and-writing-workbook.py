import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
df = pd.DataFrame({'Apples' : [30], 'Bananas' : [21]})
df.describe()
check_q1(df)
import pandas as pd

df = pd.DataFrame({'Apples' : [35, 41], 'Bananas' : [21, 34]}, index=['2017 Sales', '2018 Sales'])
print(df.head())
check_q2(df)
import pandas as pd
s = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
print(s)
check_q3(s)
import pandas as panda

data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
df = pd.DataFrame(data)
check_q4(df)
import pandas as pd
df = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name="Pregnant Women Participating")
check_q5(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
import pandas as pd
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
import pandas as pd
import sqlite3 as sql
connection = sql.connect("../input/pitchfork-data/database.sqlite")
df = pd.read_sql_query("SELECT * FROM artists", connection)
check_q7(df)