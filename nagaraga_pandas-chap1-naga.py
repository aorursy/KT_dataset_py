import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data = [[30, 21]]
fruits = pd.DataFrame(data, columns=['Apples', 'Bananas'])
#check_q1(fruits)
#print(fruits)
data = [[35, 21], [41, 34]]
sales = pd.DataFrame(data, index=['2017 Sales', '2018 Sales'], columns=['Apples', 'Bananas'])
#check_q2(sales)
#print(sales)
data = ['4 cups', '1 cup', '2 large', '1 can']
cook = pd.Series(data, index=['Flour', 'Milk', 'Eggs', 'Spam'])
#check_q3(cook)
#print(cook)
wine = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
#wine.head()
#check_q4(wine)
women = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name='Pregnant Women Participating')
#women.head()
#check_q5(women)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
#check_q6()
import sqlite3
cnx = sqlite3.connect('../input/pitchfork-data/database.sqlite')

df = pd.read_sql_query("SELECT * FROM artists", cnx)
#check_q7(df)
#df.head()