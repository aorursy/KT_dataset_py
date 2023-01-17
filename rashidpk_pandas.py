# importing Data Frame libraries 

import pandas as pd

import numpy as np
# creating data Frame 

df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['row1','row2','row3','row4','row5'],columns=['column1', 'column2', 'column3' ,'column4'])
df.head()
# saving data into Excel /csv file 

df.to_csv('test1.csv')
df['column1']
#How to access the Element

# 1. loc   2.iloc

df.loc['row1']
df.iloc[0:3,0:2]
#how to check Unique values in columns 

df['column1'].unique()
# converting data frme to arrays 

df.iloc[ :,0:].values
# calculating the null values in columns .

df.isnull().sum()


df['column1'].value_counts()
df1=pd.read_csv('D:\Machine Learning\Test.csv')
df1
df1.info()
df1.describe()
df1['X0'].value_counts()
df3=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header =None)
df3


from io import StringIO, BytesIO
# convert json to csv

df3.to_csv('win.csv')
df3
#convert json to different json formats 

df3.to_json(orient="index")
url='https://www.fdic.gov/bank/individual/failed/banklist.html'

dfs=pd.read_html(url)
dfs[0]
url2='https://en.wikipedia.org/wiki/Mobile_country_code'

dfss=pd.read_html(url2,match='Country',header=0)
dfss[0]