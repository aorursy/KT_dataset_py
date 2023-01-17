import numpy as np

import pandas as pd
labels = ['a','b','c']

my_list = [10,20,30]

arr = np.array([10,20,30])

d = {'a':10,'b':20,'c':30}
pd.Series(data=my_list)
pd.Series(data=my_list,index=labels)
pd.Series(my_list,labels)
pd.Series(arr)
pd.Series(arr,labels)
pd.Series(d)
pd.Series(data=labels)
# Even functions (although unlikely that you will use this)

pd.Series([sum,print,len])
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])                                   
ser1
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])                                   
ser2
ser1['USA']
ser1 + ser2
from numpy.random import randn
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
df
df['W']
# Pass a list of column names

df[['W','Z']]
type(df['W'])
df['new'] = df['W'] + df['Y']
df
df.drop('new',axis=1)
# Not inplace unless specified!

df
df.drop('new',axis=1,inplace=True)
df
df.drop('E',axis=0)
df.loc['A']
df.iloc[2]
df.loc['B','Y']
df.loc[['A','B'],['W','Y']]
df
df>0
df[df>0]
df[df['W']>0]
df[df['W']>0]['Y']
df[df['W']>0][['Y','X']]
df[(df['W']>0) & (df['Y'] > 1)]
df
# Reset to default 0,1...n index

df.reset_index()
newind = 'CA NY WY OR CO'.split()
df['States'] = newind
df
df.set_index('States')
df
df.set_index('States',inplace=True)
df
# Index Levels

outside = ['G1','G1','G1','G2','G2','G2']

inside = [1,2,3,1,2,3]

hier_index = list(zip(outside,inside))

hier_index = pd.MultiIndex.from_tuples(hier_index)
hier_index
df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])

df
df.loc['G1']
df.loc['G1'].loc[1]
df.index.names

df.index.names = ['Group','Num']
df
df.xs('G1')
df.xs(['G1',1])
df.xs(1,level='Num')
df = pd.DataFrame({'A':[1,2,np.nan],

                  'B':[5,np.nan,np.nan],

                  'C':[1,2,3]})
df
df.dropna()
df.dropna(axis=1)
df.dropna(thresh=2)
df.fillna(value='FILL VALUE')
df['A'].fillna(value=df['A'].mean())


# Create dataframe

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],

       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],

       'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
df
df.groupby('Company')
by_comp = df.groupby("Company")
by_comp.mean()
df.groupby('Company').mean()
by_comp.std()
by_comp.min()
by_comp.max()
by_comp.count()
by_comp.describe()
by_comp.describe().transpose()
by_comp.describe().transpose()['GOOG']
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],

                        'B': ['B0', 'B1', 'B2', 'B3'],

                        'C': ['C0', 'C1', 'C2', 'C3'],

                        'D': ['D0', 'D1', 'D2', 'D3']},

                        index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],

                        'B': ['B4', 'B5', 'B6', 'B7'],

                        'C': ['C4', 'C5', 'C6', 'C7'],

                        'D': ['D4', 'D5', 'D6', 'D7']},

                         index=[4, 5, 6, 7]) 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],

                        'B': ['B8', 'B9', 'B10', 'B11'],

                        'C': ['C8', 'C9', 'C10', 'C11'],

                        'D': ['D8', 'D9', 'D10', 'D11']},

                        index=[8, 9, 10, 11])
df1
df2
df3
pd.concat([df1,df2,df3])
pd.concat([df1,df2,df3],axis=1)
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                     'A': ['A0', 'A1', 'A2', 'A3'],

                     'B': ['B0', 'B1', 'B2', 'B3']})

   

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                          'C': ['C0', 'C1', 'C2', 'C3'],

                          'D': ['D0', 'D1', 'D2', 'D3']})  
left
right
pd.merge(left,right,how='inner',on='key')
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],

                     'key2': ['K0', 'K1', 'K0', 'K1'],

                        'A': ['A0', 'A1', 'A2', 'A3'],

                        'B': ['B0', 'B1', 'B2', 'B3']})

    

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],

                               'key2': ['K0', 'K0', 'K0', 'K0'],

                                  'C': ['C0', 'C1', 'C2', 'C3'],

                                  'D': ['D0', 'D1', 'D2', 'D3']})
pd.merge(left, right, on=['key1', 'key2'])
pd.merge(left, right, how='outer', on=['key1', 'key2'])
pd.merge(left, right, how='right', on=['key1', 'key2'])
pd.merge(left, right, how='left', on=['key1', 'key2'])
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],

                     'B': ['B0', 'B1', 'B2']},

                      index=['K0', 'K1', 'K2']) 



right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],

                    'D': ['D0', 'D2', 'D3']},

                      index=['K0', 'K2', 'K3'])
left.join(right)
left.join(right, how='outer')
import pandas as pd

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

df.head()
df['col2'].unique()
df['col2'].nunique()
df['col2'].value_counts()
#Select from DataFrame using criteria from multiple columns

newdf = df[(df['col1']>2) & (df['col2']==444)]
newdf
def times2(x):

    return x*2
df['col1'].apply(times2)
df['col3'].apply(len)
df['col1'].sum()
del df['col1']
df
df.columns
df.index
df
df.sort_values(by='col2') #inplace=False by default
df.isnull()
# Drop rows with NaN Values

df.dropna()
import numpy as np
df = pd.DataFrame({'col1':[1,2,3,np.nan],

                   'col2':[np.nan,555,666,444],

                   'col3':['abc','def','ghi','xyz']})

df.head()
df.fillna('FILL')
data = {'A':['foo','foo','foo','bar','bar','bar'],

     'B':['one','one','two','two','one','one'],

       'C':['x','y','x','y','x','y'],

       'D':[1,3,2,5,4,1]}



df = pd.DataFrame(data)
df
df.pivot_table(values='D',index=['A', 'B'],columns=['C'])
import numpy as np

import pandas as pd
#df = pd.read_csv('the name of you .csv file')

#df
df.to_csv('example',index=False)
#pd.read_excel('Excel_Sample.xlsx',sheetname='Sheet1')
df.to_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
# df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df.to_sql('data', engine)
sql_df = pd.read_sql('data',con=engine)
sql_df