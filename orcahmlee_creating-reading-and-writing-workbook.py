import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
df = pd.DataFrame({'Apples' : [30],'Bananas' : [21]})
check_q1(df)
df = pd.DataFrame(data={'Apples':[35,41], 'Bananas':[21,34]}, index=['2017 Sales', '2018 Sales'])
check_q2(df)
Dinner = pd.Series(data=['4 cups', '1 cup', '2 large', '1 can'], index=['Flour' ,'Milk' , 'Eggs', 'Spam'])
check_q3(Dinner)
data = pd.read_csv(filepath_or_buffer='../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
data.head()
check_q4(data)
data = pd.read_excel(
    '../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',
    sheet_name='Pregnant Women Participating'
)
data.head()
check_q5(data)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv('cows_and_goats.csv'))
import sqlite3
conn = sqlite3.connect('../input/pitchfork-data/database.sqlite')
data = pd.read_sql(
    sql = 'select * from artists',
    con = conn
)
data.head()
check_q7(data)