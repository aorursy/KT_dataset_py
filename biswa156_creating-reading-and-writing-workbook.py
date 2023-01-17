import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
import pandas as pd
check_q1(pd.DataFrame({'Apples':[30],'Bananas':[21]}))

# Your code here
import pandas as pd
check_q2(pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales']))
# Your code here
import pandas as pd
check_q3(pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name='Dinner'))
# Your code here 
import pandas as pd
df=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
df=df.iloc[:,1:]
check_q4(df.head())
# Your code here
import pandas as pd
check_q5(pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating'))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df)
# Your code here
import pandas as pd
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
import pandas as pd
import sqlite3 as sql
conn=sql.connect('../input/pitchfork-data/database.sqlite')
check_q7(pd.read_sql('SELECT * FROM artists;',conn))
