import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
a = {'Apples':[30],'Bananas':[21]}
df=pd.DataFrame(a)
df
check_q1(df)
# Your code here
import pandas as pd
a = {'Apples':[35,41],'Bananas':[21,34]}
df=pd.DataFrame(a,index=['2017 Sales','2018 Sales'])
df
check_q2(df)
# Your code here
import pandas as pd
a = ['4 cups','1 cup','2 large','1 can']
df=pd.Series(a,name='Dinner',index=['Flour','Milk','Eggs','Spam'])
df
check_q3(df)
# Your code here 
import pandas as pd
df=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
df=df.drop('Unnamed: 0',axis=1)
df
check_q4(df)
# Your code here
import pandas as pd
pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls','Pregnant Women Participating')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3
df = pd.read_sql('select * from artists',con=sqlite3.connect('../input/pitchfork-data/database.sqlite'))
df
check_q7(df)
