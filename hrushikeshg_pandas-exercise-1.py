import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())

dataframe=pd.DataFrame({'Apples':[30],'Bananas':[21]})
print (dataframe)
da=pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
print(da)
labe=['Flour','Milk','Eggs','Spam']
qty=['4 cups','1 cup','2 large','1 can']

df2=pd.Series(data=qty,index=labe)
print(df2)
df3=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
print(df3)
df=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
print(df.head(20))
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv",index=False)

from sqlalchemy import create_engine
engine = create_engine('sqlite://input//pitchfork-data//database.sqlite')
sqld=pd.read_sql('artists',con=engine)