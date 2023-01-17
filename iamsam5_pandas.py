import pandas as pd

import numpy as np
l=[10,20,30,40,50]

s1=pd.Series(l) #[parameter1-Data] [Parameter2-index, default -0,1..]

s1
myd={'a':10,'b':20,'c':30}

s2=pd.Series(myd) #keys->index values->data

s2
#Accessing - using index

myd['a']
#changing any data using its key 

s2['a']=100
s2
#adding new enties

s2['d']=40
s2
#Deleting entry

del s2['d']
s2
s1
s3=pd.Series([0,1,2,3,4],[1,2,3,4,5])
s3
s1+s3 #if n=index does not match then nan, if it matches then it performs operation.
df=pd.DataFrame(np.random.randn(5,4),'1 2 3 4 5'.split(),'A B C D'.split())

#para 1 - data 

#para 2 - index

#para 3 - column
df
#Selecting column

#Selcting single column

df['A']

#Selecting more than 1 column

df[['A','B']]
df
#Selceting rows using iloc[] and loc[]

#selecting single row

df.loc['1']
#Selecting more than 1 row

df.loc[['1','2']]
#Selcting an element

df.loc['1','A']
#Using iloc[] i- index

df.iloc[0]
df
#Creating new column 

df['E']=df['A']+df['B']
df
df['F']=[1,2,3,4,5]
df
#Deleting column

df.drop('F',axis=1,inplace=True)
df

df.drop('5',axis=0,inplace =True)
df
df>True
df[df>True]
[df['A']>0]
df[df['A']>0]
df[df['A']>0]['B'] # when the fisrt conditon is true and then the column b elements are selected
df[(df['A']>0) & (df['B'] > 1)]



#&-and operation

#| - or operation
df
df.index="10 20 30 40 ".split()
df
#using an existing column to use it as its index

df.set_index('E')



#inplace needs to be passed for permannet change
df
df.reset_index()
df = pd.DataFrame({'A':[1,2,np.nan],

                  'B':[5,np.nan,np.nan],

                  'C':[1,2,3]})
df
df.dropna(axis=1,inplace=True)

#Deletes the row which has nan value 

#axis=0 -> row

#axis=1 -> column

df1 = pd.DataFrame({'A':[1,2,np.nan],

                  'B':[5,np.nan,np.nan],

                  'C':[1,2,3]})
df1
df1.fillna(10)

#Replaces the nan with the value passed 
df
op=pd.DataFrame({'a':[1,2,3,4,5],'b':[11,22,33,44,55]})
op

op['a'].unique()
op['b'].nunique()
def times2(x):

    return x*3
op['b'].apply(times2)
op['b'].sum()
op['b'].min()

op.sort_values(by='a')
df.isnull()
df.dropna()