# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#series
obj=pd.Series([4,7,-5,3])
obj
obj.values
obj.index
obj2=pd.Series([4,-7,5,3],index=['d','b','a','c'])
obj2
obj2.index
obj2['a']
obj2['d']=6
obj2[['c','a','d']]
obj2[obj2>0]
#scalar multiplication
obj2*2
#applying math function
np.exp(obj2)
'b' in obj2
'e' in obj2
sdata={'ohio':3500,'texas':7100,'oregon':1600,'utah':500}
obj3=pd.Series(sdata)
obj3
obj3.index
states=['california','ohio','oregon','utah']
obj4=pd.Series(sdata,index=states)
obj4
pd.isnull(obj4)
pd.notnull(obj4)
obj3
obj4
obj3+obj4
obj4.name='population'
obj4.index.name='state'
obj4
obj
obj.index=['bob','steve','jeff','ryan']
obj
#dataframe
data={'state':['ohio','ohio','ohio','nevada','nevada','nevada'],
     'year': [2000,2001,2002,2001,2002,2003],
     'population':[1.5,1.7,3.6,2.4,2.8,3.2]}
frame=pd.DataFrame(data)
frame
frame.head()
pd.DataFrame(data,columns=['year','state','population'])
frame2=pd.DataFrame(data,columns=['year','state','population','debt'],index=['one','two','three','four','five','six'])
frame2
frame2.columns
frame2['state']
frame.year
frame2.year
frame2['debt']=16.5
frame2
frame2['debt']=np.arange(1,13,2)
frame2
val1=[3,2,4,8,7,6]
frame2['debt']=val1
frame2
val2=pd.Series([-1.2,-1.5,-1.7],index=['two','four','five'])
frame2['debt']=val2
frame2
frame2['eastern']=frame2.state=='ohio'
frame2
del frame2['eastern']
frame2.columns
pop={'nevada':{2001:2.4,2002:2.9},
    'ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3=pd.DataFrame(pop)
frame3
frame3.T
frame3.index.name='year';frame3.columns.name='state'
frame3
frame3.values
frame2.values
#index objects
obj=pd.Series(range(3),index=['a','b','c'])
index=obj.index
index
index[1:]
index[1]='d'
labels=pd.Index(np.arange(3))
labels
obj2=pd.Series([1.5,-2.5,0],index=labels)
obj2
obj2.index is labels
dup_labels=pd.Index(['foo','foo','bar','bar'])
dup_labels
#essentialfunctionality
#dropping entries from axis
obj=pd.Series(np.arange(5.),index=['a','b','c','d','e'])
obj
new_obj=obj.drop('c')
new_obj
obj.drop(['d','c'])
data=pd.DataFrame(np.arange(16).reshape((4,4)),
                 index=['ohio','colorado','utah','new york'],
                 columns=['one','two','three','four'])
data
data.drop(['colorado','ohio'])
data.drop('two',axis=1)
data.drop(['two','four'],axis='columns')
obj.drop('c',inplace=True)
obj
#indexing,selection and filtering
obj=pd.Series(np.arange(4.),index=['a','b','c','d'])
obj
obj['b']
obj[1]
obj[2:4]
obj[['b','a','d']]
obj[[1,3]]
obj[obj<2]
obj['b','c']=5
obj
data=pd.DataFrame(np.arange(16).reshape((4,4)),
                 index=['ohio','colorado','utah','new york'],
                 columns=['one','two','three','four'])
data
data.T
data['two']
data[['three','one']]
data[data['three']>5]
data<5
data[data<5]=0
data
#selection with loc and iloc
data.loc['colorado',['two','three']]
data.iloc[2,[3,0,1]]
data.iloc[2]
data.iloc[[1,2],[3,0,1]]
data.loc[:'utah','two']
data.iloc[:,:3][data.three>5]
#arithmetic and data alignment
s1=pd.Series([7.3,-2.5,3.5,1.5],index=['a','c','d','e'])
s2=pd.Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])
s1
s2
s1+s2
df1=pd.DataFrame(np.arange(9.).reshape((3,3)),columns=list('bcd'),index=['ohio','texas','colorado'])
df2=pd.DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index=['utah','ohio','texas','oregon'])
df1
df2
df1+df2
#arithmetic methods with fill values
df1=pd.DataFrame(np.arange(12.).reshape((3,4)),columns=list('abcd'))
df2=pd.DataFrame(np.arange(20.).reshape((4,5)),columns=list('abcde'))
df2.loc[1,'b']=np.nan
df1
df2
df1+df2
df1.add(df2,fill_value=0)
1/df1
df1.rdiv(1)
#operations between dataframe adn series
arr=np.arange(12.).reshape((3,4))
arr
arr[0]
arr-arr[0]
frame=pd.DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index=['utah','ohio','texas','oregon'])
series=frame.iloc[0]
frame
series
frame-series
series2=pd.Series(range(3),index=['b','e','f'])
frame+series2
frame.add(series2)
series3=frame['d']
frame
series3
frame.sub(series3,axis='index')
#function application mapping
frame=pd.DataFrame(np.random.randn(4,3),columns=list('bcd'),
                  index=['utah','ohio','texas','oregon'])
frame
np.abs(frame)
f=lambda x: x.max()-x.min()
frame.apply(f)
frame.apply(f,axis='columns')
frame.sum()
frame.mean()
frame.mean(axis='columns')
def f(x):
    return pd.Series([x.min(),x.max()],index=['min','max'])
frame.apply(f)
format=lambda x: '%.2f'%x
frame.applymap(format)
frame['d'].map(format)
obj=pd.Series(range(4),index=['d','a','b','c'])
obj
obj.sort_index()
frame=pd.DataFrame(np.arange(8.).reshape((2,4)),index=['two','three'],columns=list('dabc'))
frame
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1,ascending=False)
obj=pd.Series([4,7,-3,2])
obj.sort_values
obj=pd.Series([4,np.nan,7,np.nan,-3,2])
obj.sort_values()
frame=pd.DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
frame
frame.sort_values(by='b')
frame.sort_values(by=['a','b'])
#summarizing and computing descriptive statistics
df=pd.DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],
               columns=['one','two'])
df
df.sum(axis=1)
df.sum()
df.sum(axis='columns')
df.mean(axis='columns',skipna=False)
df.idxmax()
df
df.cumsum()
df.describe()
obj=pd.Series(['a','a','b','c']*4)
obj.describe()
#unique values,value counts,and membership
obj=pd.Series(['c','a','d','a','a','b','b','c','c'])
obj.value_counts()
obj
mask=obj.isin(['b','c'])
mask
obj[mask]
daya=pd.DataFrame({'qu1':[1,3,4,3,4],
                  'qu2':[2,3,1,2,3],
                  'qu3':[1,5,2,4,4]})
data
daya
result=daya.apply(pd.value_counts).fillna(0)
result
