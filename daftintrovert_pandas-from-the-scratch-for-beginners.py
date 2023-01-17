#series intro

import numpy as np

import pandas as pd
label = ['a','b','c']

my_data = [10,20,30]

arr = np.array(my_data)

d = {'a':10,'b':20,'c':30}
pd.Series(data = my_data,index = label)
pd.Series(arr,label)
ser1 = pd.Series([1,2,3,4],['India','USSR','China','Germany'])
ser1
ser2 = pd.Series([1,2,5,4],['India','USSR','Italy','Germany'])
ser2
ser1+ser2
#DataFrames



from numpy.random import randn
np.random.seed(101)
df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
df
df['W']
type(df['W'])
type(df)
df.tail(1)
df.mean()
df[['W','Z']]
df['New'] = df['W'] + df['X']
df
df.drop('New', axis = 1,inplace = True)
df
df.drop('E')
df.shape
df
df[['X','Y']]
#SELECTING ROWS

df
df.loc['A'] #first method using loc
df.iloc[2] #index based location
df.loc['A','X'] #Finding subsets
df.loc[['A','D'],['Y','Z']]
#DataFrame Part 2
df >0
booldf = df>0
booldf
df[booldf]
df
df[df['W']>0]['Y']
df[(df['W']>0) & (df['Z']>0)] # & is the replacement for and as it will give error if we use 'and' (ampersend)
df[(df['W']>0) | (df['Y']>0)] # | is the replacement for or as it will give error if we use "or" (pipe_operator)
df.reset_index() # use inplace = True if you want to make this change permanent
df
newind = 'BR MP TN DL HR'.split() #shortcut to add a column in the dataset
newind
df['States'] = newind
df
df.drop(['States'],axis = 1) #for dropping the column
df
df.set_index('States') #changing it into index
#data frame part 3



#reading about multilayer and layer hierachy
import numpy as np

import pandas as pd
# Index levels

outside = ['G1','G1','G1','G2','G2','G2']

inside = [1,2,3,1,2,3]

hier_index = list(zip(outside, inside))

hier_index = pd.MultiIndex.from_tuples(hier_index)
outside
inside
list(zip(outside, inside))
df = pd.DataFrame(randn(6,2),hier_index,['A','B'])
df
df.loc['G1'].loc[1]
df.index.names = ['Groups','Nums']
df

df.loc['G2'].loc[2].loc["A"] 
#cross section is used only when you have a multi lavel index and is denoted by xs

df
#we want to grab everything under G1

df.xs('G1')
#missing data

d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}
df = pd.DataFrame(d)
df
df.dropna()
df.dropna(axis = 1) #for dropping columns
df
df.dropna(thresh = 2) #minimun Nan in the column
df

df.fillna(value = 'fill') #filling the nan space with our values
#filling the column with the mean of the column



df['A'].fillna(value = df['A'].mean())
#groupby



#groupby allows you to group together rows based off a column and perform an aggregate function on them
data = {'Company':['GOOG','GOOG','FB','FB','MSFT','MSFT'],'Person':['Sam','Duke','Amy','Rach','Dora','Mike'],'Sales':[200,120,420,374,412,358]}
data
df = pd.DataFrame(data)
df
df.groupby('Company')
byComp = df.groupby('Company')
byComp
byComp.mean()
byComp.std()
byComp.sum().loc['FB']
byComp.count()
byComp.describe()
byComp.describe().transpose()