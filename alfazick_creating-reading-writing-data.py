import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here


df1 = pd.DataFrame.from_dict({ 0: {"Apples":30, "Bananas":21}}, orient='index')
    
check_q1(df1)
# Your code here

df2 = pd.DataFrame.from_dict({ '2017 Sales': {"Apples":35, "Bananas":21},
                               '2018 Sales': {"Apples":41, "Bananas":34}},
                             orient='index')


check_q2(df2)
# Your code here
ans3 = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour','Milk','Eggs','Spam'], name='Dinner')


check_q3(ans3)

# Your code here 
filename = '../input/wine-reviews/winemag-data_first150k.csv'

ans4 = pd.read_csv(filename)
ans4 = ans4.drop(ans4.columns[0], axis = 1)

check_q4(ans4)
import warnings
warnings.filterwarnings("ignore")
# Your code here
file_name = '../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls'
ans5 = pd.read_excel(file_name, sheet_name = 'Pregnant Women Participating')
check_q5(ans5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
check_q6()
# Your Code Here
import sqlite3
import pandas as pd
# Create your connection.
cnx = sqlite3.connect('../input/pitchfork-data/database.sqlite')
df = pd.read_sql_query("SELECT * FROM artists", cnx)
check_q7(df)