import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
pd.DataFrame({'Apples': [30], 'Bananas': [21]})
check_q1(pd.DataFrame())
print(answer_q1())
import pandas as pd
pd.DataFrame({'Apples' : [35,41], 
              'Bananas': [21,34]},
             index =['2017 Sales','2018 Sales'] )
#check_q2(pd.DataFrame())
#print(answer_q2())
import pandas as pd 
s = pd.Series(['4 cups','1 cup','2 large','1 can'],index = ['Flour','Milk','Eggs','Spam'], name='Product A')
check_q3(s)                                                                                                                        
import pandas as pd
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
check_q4(wine_reviews)

import pandas as pd
wic = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", 
                    sheet_name='Pregnant Women Participating')
wic.head()
check_q5(wic)

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
import pandas as pd
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
check_q6(q6_df)

import sqlite3
import pandas as pd
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
fires = pd.read_sql_query("SELECT * FROM artists", conn)
fires.head()
print(fires)
check_q7(fires)