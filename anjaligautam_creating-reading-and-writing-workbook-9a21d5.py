import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
d = {'Apples':[30],'Bananas':[21]}
x=pd.DataFrame(data=d)
print(x)
check_q1(x)
# Your code here
d1 = {'Apples':[35,41],'Bananas':[21,34]}
x1 = pd.DataFrame(data=d1,index=['2017 Sales','2018 Sales'])
print(x1)
check_q2(x1)
# Your code here
d2={'Flour':'4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}
x2 = pd.Series(data=d2,name='Dinner',index=['Flour','Milk','Eggs','Spam'])
print(x2)
check_q3(x2)
# Your code here 
x4=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
x4.head()
check_q4(x4)
# Your code here
x5=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
check_q5(x5)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
disc=q6_df.to_csv('cows_and_goats.csv')
check_q6(disc)
# Your Code Here