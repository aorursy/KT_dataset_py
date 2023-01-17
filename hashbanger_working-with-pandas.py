import numpy as np
import pandas as pd
data = [10,20,30]
labels = ['a','b','c']
arr = np.array(data)
adict = {'a': 10, 'b':20, 'c': 40}
pd.Series(data = data, index = labels)
pd.Series(data, labels)
pd.Series(arr, labels)
pd.Series(adict)
pd.Series([print,sum,max])
f = pd.Series([print,sum,max])
f[0]("Hello")
ser1 = pd.Series( [1,2,3,5],['USA','Japan','USSR','Germany'])
ser2 = pd.Series([6,7,4,2],['USA','Italy','Germany','USSR'])
ser1 + ser2
df = pd.DataFrame(np.linspace(0,1,25).reshape(5,5),['A','B','C','D','E'],['V','W','X','Y','Z'])
df
df['W']  #or df.W  however not recommended
print(type(df))
print(type(df['W']))
df[['W','Y']]
df['NEW'] = df['W']+df['Y']
df
df.drop('NEW', axis = 1, inplace = True)
#or
#df = df.drop('NEW', axis = 1)
df
df.shape 
df.loc[['A','B']]
df.iloc[1]
df = pd.DataFrame(np.random.randn(25).reshape(5,5),['A','B','C','D','E'],['V','W','X','Y','Z'])
df
df[ df < 0.5]
df[df['V'] < 0]
df[df['V'] < 0][['W','X']]
df[(df['V'] < 0) & (df['W'] > 0)]
df[(df['W'] < 0) | (df['X'] > 1)]
df.reset_index()
states = 'CO NY WY OK CH'.split()
df['states'] = states
df
df.set_index(df['states']).drop('states', axis = 1)
outside = ['G1','G1','G1','G2','G2','G2']
inside  =  [1,2,3,1,2,3]
hier_index  = list(zip(outside, inside))
print(hier_index)
hier_index  = pd.MultiIndex.from_tuples(hier_index)
hier_index
ddf = pd.DataFrame(np.random.randn(6,2),hier_index, ['A','B'])
ddf
ddf.index.names = ['Groups', 'Nums']
ddf
ddf.loc['G2'].loc[3]['B']
daf  = {'A': [1, 2, np.nan], 'B': [4, np.nan, np.nan], 'C': [7,8,9]}
daf = pd.DataFrame(daf)
daf
daf.dropna()
daf.dropna(axis = 1)
daf.dropna(thresh = 2)
daf.fillna(value = 'FILLED')
daf['A'].fillna(value = (daf['A'].mean()), inplace = True)
daf['B'].fillna(value = (daf['B'].mean()), inplace = True)
daf
sa = pd.DataFrame({'Sales': [450,120,345,334,232,100],'Person': ['Prashant','Shivam','Shiva','Ankit','Arpit','Abhi']
                  ,'Company':['Microsoft','Microsoft','Google','Google','Apple','Apple']
                   })
sa
byComp = sa.groupby('Company')
byComp
byComp.mean()
byComp.median()
byComp.median().loc['Microsoft']
sa.groupby('Company').median().loc['Microsoft']
sa.groupby('Company').count()
sa.groupby('Company').max()
sa.groupby('Company').min()
sa.groupby('Company').describe().transpose()
sa.groupby('Company').describe().transpose()['Microsoft']
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
pd.concat([df1, df2, df3])
pd.concat([df1, df2, df3], axis = 1)
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})    
left
right
pd.merge(left, right, how = 'inner', on = 'key')
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})
left
right
pd.merge(left, right, on = ['key1', 'key2'])
pd.merge(left, right, how = 'outer', on = ['key1','key2'])
pd.merge(left, right, how = 'left', on = ['key1', 'key2'])
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])
left.join(right)
left.join(right, how = 'outer')
import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df
df['col2'].unique()
df['col2'].nunique() #equivalent to using len(df['col2'].unique())
df['col2'].value_counts()
df[(df['col1'] > 1) & (df['col2'] == 444)]
def times2(n):
    return n*2
df['col2'].apply((lambda n: n*2)) #or df['col2'].apply(times2)
df.columns
df.index
df.sort_values('col2', ascending = False)
df.isnull()
data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)
df
df.pivot_table(values = 'D', index = ['A','B'], columns = 'C')
dff = pd.read_csv('../input/example')
dff
dff.to_csv('ToCSVoutput.csv', index = False)
pd.read_csv('ToCSVoutput.csv')
ddff = pd.read_excel('../input/Excel_Sample.xlsx', sheet_name= 'Sheet1')
ddff
ddff.to_excel('ToXLoutput.xlsx', sheet_name= 'Sheet1')