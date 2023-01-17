import numpy as np

import pandas as pd
labels = ['a','b','c']

my_list = [10,20,30]

arr = np.array([10,20,30])

d = {'a':10,'b':20,'c':30}
# using list

pd.Series(data=my_list)
# using list with custom index 

pd.Series(data=my_list,index=labels)
# without using assignment operator

pd.Series(my_list,labels)
# using a numpy array

pd.Series(arr)
# using a numpy array with indexes

pd.Series(arr,labels)
# using a dictionary

pd.Series(d)
# Even functions (although unlikely that you will use this)

pd.Series([sum,print,len])
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])  

ser1
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])

ser2
ser1 + ser2
from numpy.random import randn

np.random.seed(101)
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())

df
df['W']
# Pass a list of column names

df[['W','Z']]
df['new'] = df['W'] + df['Y']

df
df.drop('new',axis=1)

# Not inplace unless specified!

df
# inplace will save the changes in the dataframe

df.drop('new',axis=1,inplace=True)

df
# drop horizontal axis

df.drop('E',axis=0)
# Selecting Rows

df.loc['A']
# select based off of position instead of label

df.iloc[2]
# selecting subset of rows and columns

df.loc['B','Y']
df.loc[['A','B'],['W','Y']]
df
# show the boolean value where the dataframe satisfies

df>0
# shows the cell value where condition is satisfied

df[df>0]
# shows the rows for which the condition is satisfied

df[df['W']>0]
df[df['W']>0]['Y']
df[df['W']>0][['Y','X']]
# For two conditions you can use | and & with parenthesis:

df[(df['W']>0) & (df['Y'] > 1)]
newind = 'CA NY WY OR CO'.split()
# adding a new row to the dataframe

df['States'] = newind

df
# showing/letting the index with the desired column

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
df = pd.DataFrame({'A':[1,2,np.nan],

                  'B':[5,np.nan,np.nan],

                   'C':[1,2,3]})

df
# drop the rows where its NaN

df.dropna()
# drop the column where its NaN

df.dropna(axis=1)
# fill with desired value the rows where its NaN

df.fillna(value='FILL VALUE')
df['A'].fillna(value=df['A'].mean())
df
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],

       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],

       'Sales':[200,120,340,124,243,350]}

data
df = pd.DataFrame(data)

df
df.groupby('Company')
# You can save this object as a new variable:

by_comp = df.groupby("Company")
# call aggregate methods off the object:

by_comp.mean()
by_comp.std()
by_comp.min()
by_comp.max()
by_comp.count()
by_comp.describe()
by_comp.describe().transpose()['GOOG']
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],

                        'B': ['B0', 'B1', 'B2', 'B3'],

                        'C': ['C0', 'C1', 'C2', 'C3'],

                        'D': ['D0', 'D1', 'D2', 'D3']},

                        index=[0, 1, 2, 3])

df1
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],

                        'B': ['B4', 'B5', 'B6', 'B7'],

                        'C': ['C4', 'C5', 'C6', 'C7'],

                        'D': ['D4', 'D5', 'D6', 'D7']},

                         index=[4, 5, 6, 7]) 

df2
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],

                        'B': ['B8', 'B9', 'B10', 'B11'],

                        'C': ['C8', 'C9', 'C10', 'C11'],

                        'D': ['D8', 'D9', 'D10', 'D11']},

                        index=[8, 9, 10, 11])

df3
pd.concat([df1,df2,df3])
pd.concat([df1,df2,df3],axis=1)
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                     'A': ['A0', 'A1', 'A2', 'A3'],

                     'B': ['B0', 'B1', 'B2', 'B3']})

   

left
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                          'C': ['C0', 'C1', 'C2', 'C3'],

                          'D': ['D0', 'D1', 'D2', 'D3']})

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
pd.merge(left, right, how='left', on=['key1', 'key2'])
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],

                     'B': ['B0', 'B1', 'B2']},

                      index=['K0', 'K1', 'K2']) 



right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],

                    'D': ['D0', 'D2', 'D3']},

                      index=['K0', 'K2', 'K3'])
left.join(right)
left.join(right, how='outer')
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

df.head()
df['col2'].unique()
df['col2'].nunique()
df['col2'].value_counts()
# Select from DataFrame using criteria from multiple columns

newdf = df[(df['col1']>2) & (df['col2']==444)]

newdf
# Applying Functions

def times2(x):

    return x*2



df['col1'].apply(times2)
df['col3'].apply(len)
df['col1'].sum()
df.columns
df.index
df.sort_values(by='col2') #inplace=False by default
df.isnull()
# Drop rows with NaN Values

df.dropna()