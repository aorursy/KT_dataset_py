import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
import pandas as pd
df = pd.DataFrame({'Apples':[30],'Bananas':[21]})
df

# Your code here
pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
# Your code here
Dinner={'Flour':'4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}
df=pd.Series(Dinner)
df.name='Dinner'
df
# Your code here 
url='../input/wine-reviews/winemag-data_first150k.csv'
df=pd.read_csv(url)
dat=pd.DataFrame(df)
dat
# Your code here
url='../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls'
xls=pd.read_excel(url)
df=pd.DataFrame(xls)
df

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here