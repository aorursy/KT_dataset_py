import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
data={'Apples':[30],'Bananas':[21]}
df=pd.DataFrame(data, columns=['Apples','Bananas'])
#check_q1()# Your code here

#import pandas as pd
#pd.set_option('max_rows', 5)

#import sys
#sys.path.append('../input/advanced-pandas-exercises/')
#from creating_reading_writing import *
check_q1(df)
#print(answer_q1())
# Your code herimport pandas as pd
data={'Apples':[35,41],'Bananas':[21,34]}
df2=pd.DataFrame(data, columns=['Apples','Bananas'],index=['2017 Sales','2018 Sales'])
check_q2(df2)
print(answer_q2())
Dinner =pd.Series (['4 cups', '1 cup', '2 large', '1 can'], index=['Flour','Milk','Eggs','Spam']) 
check_q3(Dinner)
print(answer_q3())
df4=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)#index=['country', 'description','points','price','province','region_1','region_2','variety','winery']
#df4.info()
check_q4(df4)
#print(answer_q4())

ex=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
check_q5(ex)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
#check_q6(g)
print(answer_q6())

import sqlite3
conn=sqlite3.connect("../input/pitchfork-data/database.sqlite")
p=pd.read_sql_query("Select * from artists",conn)
check_q7(p)