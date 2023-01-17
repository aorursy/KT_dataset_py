import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
fruit={ 'Apples':[30],'Bananas':[21]}
df=pd.DataFrame(fruit)
df
fruit={'Apples':[35,41],'Bananas':[21,34]}
df=pd.DataFrame(fruit,index=['2017 Sales','2018 Sales'])
df
import pandas as pd
d = {'Dinner' : pd.Series(['4 cups','1 cup','2 large','1 can'], index=['Flour', 'Milk', 'Eggs','Spam'])}
df = pd.DataFrame(d)
type(df['Dinner'])
df['Dinner']
data=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
data
import pandas as pd 
data=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',"Pregnant Women Participating") 
data
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
import pandas as pd
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
import pandas as pd
import sqlite3
from os import path
db=sqlite3.connect('../input/pitchfork-data/database.sqlite')
df=pd.read_sql('select * from artists',db)
#connection.close()
df