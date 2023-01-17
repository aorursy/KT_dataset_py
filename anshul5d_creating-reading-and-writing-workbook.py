import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
import pandas as pd
data ={'Apple':[30],'Bananas':[21]}
dataframe = pd.DataFrame(data)
dataframe
# Your code here
dataframe = pd.DataFrame({'Apple': [35,41],'Bananas':[21,34]}, index=['2017 Sales','2018 Sales'])
dataframe
# Your code here
data = {'Flour':'4 cups','Milk':'1 Cups','Eggs':'2 Large','Spam':'1 Can'}
pd.Series(data).rename('Dinner')
# Your code here 
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
# Your code here
#filepath='../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls'
xl= pd.ExcelFile('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
preg = pd.read_excel(xl,'Pregnant Women Participating')
#pregna = pd.read_excel(filepath,'Pregnant Women Participating')
#Pregnant=pd.read_excel(open('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls','Pregnant Women Participating'))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3
con= sqlite3.connect('../input/pitchfork-data/database.sqlite')
device = pd.read_sql_query('select * from artists',con)
device
