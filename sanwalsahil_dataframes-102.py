import pandas as pd

obj = pd.Series([2,3,4,2])

obj
obj.values
obj.index
obj2 = pd.Series([2,3,1,4,3],index=['a','s','d','e','f'])

obj2
obj2['a']
obj2[['a','s','d']]
obj2[obj2>2]
obj2*2
import numpy as np

np.exp(obj2)
'a' in obj2
sdata = {'a':300,'b':900,'c':45}

obj3 = pd.Series(sdata)

obj3


obj = pd.Series(sdata,index = ['c','b','a'])

obj


obj = pd.Series(sdata,index = ['c','b','a','e'])

obj
obj.isnull()
obj.notnull()
obj1= pd.Series([2,3,2],index=['a','s','d'])

obj1
obj2= pd.Series([2,3,2],index=['s','d','a'])

obj2
obj1+obj2
obj2.name = 'o-name'

obj2.index.name = 'i-name'

obj2
obj2.index = ['sahil','deepak','amit']

obj2
obj = pd.DataFrame({

    'name':['sahil','richard','rafi'],

    'score':[1,2,3]

})

obj
obj.head(2)
obj = pd.DataFrame({

    'name':['sahil','richard','rafi'],

    'score':[1,2,3]

},columns=['name','score','Day'])

obj
obj['name']
obj.name
obj2
obj2['sahil']
obj
obj['Day'] = 'monday'

obj
obj = pd.DataFrame({

    'name':['sahil','richard','rafi'],

    'score':[1,2,3]

},columns=['name','score','Day'])

obj
obj['Day'] = ['monday']
obj['month'] = 'Feb'
obj
del obj['month']
obj
obj.T
obj.index.name = 'i-names'

obj.columns.name = 'o-names'

obj
obj.values
obj.index
obj
obj.reindex([2,1,0])
obj['Day'][1]= 'monday'
obj.reindex([2,1,0,3,4],method='ffill')
obj.reindex(columns = ['score','name','Day'])
obj
obj.drop([2,1])
obj.drop(['name','score'],axis=1)
obj
obj[0:1]
obj[0:2]['Day'] = 'wednesday'
obj
obj[['name','score']]
obj[['name','score']][0:1]
obj[obj['score']>1]
obj.loc[:,['name','score']]
s1 = pd.Series([1,2,3],index=['a','b','c'])

s2 = pd.Series([2,3,4,4,3],index=['a','b','c','d','e'])

s3 = s1+s2

s3
s3= s1.add(s2,fill_value=0)

s3
s3 = s1.add(s2,fill_value=1)

s3
obj = pd.DataFrame(np.arange(12).reshape((4,3)),

                  columns=list('bde'))

obj
se = obj.iloc[0]

se
obj-se
se2 = pd.Series(range(3),index=['b','e','f'])

se2
obj-se2