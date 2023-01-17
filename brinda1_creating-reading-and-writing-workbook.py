import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data={'Apples':[30],'Bananas':[21]}
pd.DataFrame(data)

data={'Apples':[35,41],'Bananas':[21,34]}

pd.DataFrame(data, index={'2017 Sales','2018 Sales'}) 
data=['4 cups', '1 cup','2 large','1 can']
index={'Flour','Milk','Eggs','Spam'}
pd.Series(data,index)

data=pd.read_csv( '../input/wine-reviews/winemag-data_first150k.csv')
pd.DataFrame(data)

with pd.ExcelFile('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls') as xls: 
    df1 = pd.read_excel(xls,'Pregnant Women Participating')
pd.DataFrame(df1)
 
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here