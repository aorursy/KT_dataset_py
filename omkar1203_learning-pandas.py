# importing numpy and pandas

import numpy as np

import pandas as pd
df = pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','Row5'],columns=["Column1","Column2","Column3","Column4"])
df
df.head()
df.loc['Row1']
df.to_csv('Test.csv')
df.loc['Row1']
df.iloc[:,:]
df.iloc[:,1:]
df.iloc[1:3,1:2]
df.iloc[1:3,1:2].values
df.iloc[:,1:].values
df['Column1'].value_counts()
df.iloc[:,1:].values.shape
df['Column1'].unique()
df.describe()
df.info()
from io import StringIO, BytesIO
data = ('col1,col2,col3\n'

            'x,y,1\n'

            'a,b,2\n'

            'c,d,3')
type(data)
pd.read_csv(StringIO(data))
df=pd.read_csv(StringIO(data), usecols=lambda x: x.upper() in ['COL1', 'COL3'])
df.to_csv('Test.csv')
data = ('a,b,c,d\n'

            '1,2,3,4\n'

            '5,6,7,8\n'

            '9,10,11')
print(data)
df=pd.read_csv(StringIO(data),dtype=object)
df
df['a'][1]
df=pd.read_csv(StringIO(data),dtype={'b':int,'c':np.float,'a':'Int64'})
df
df['a'][1]
Data = '{"employee_name": "James", "email": "james@gmail.com", "job_profile": [{"title1":"Team Lead", "title2":"Sr. Developer"}]}'

pd.read_json(Data)
url = 'https://www.fdic.gov/bank/individual/failed/banklist.html'



dfs = pd.read_html(url)
dfs[0]
url_mcc = 'https://en.wikipedia.org/wiki/Mobile_country_code'

dfs = pd.read_html(url_mcc, match='Country', header=0)
dfs[0]