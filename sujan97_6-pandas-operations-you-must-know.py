import numpy as np

import pandas as pd

import seaborn as sns



df = sns.load_dataset('titanic')

df.head(10)
df.describe()
df.describe(include='all')
df.info()
#Indexing titanic data with row(1,7,21,10) and column(sex,age,fare,who,class)

smallData = df.loc[[1,7,21,10], ['sex','age','fare','who','class']]

smallData
#Creating a DataFrame with exact matching columns of smallData

newData = pd.DataFrame({'sex':['female', 'male','male'], 

                        'age':[25,49,35], 

                        'fare':[89.22,70.653,30.666], 

                        'who':['child', 'women', 'man'], 

                        'class':['First','First','First']})

newData
pd.concat([smallData, newData])
pd.concat([ newData,smallData], ignore_index=True)
pd.concat([smallData, newData], axis=1)
print('-----newData------\n',newData)

print('-----smallData----\n',smallData)
newData = pd.DataFrame({'fare':[89.22,70.653,30.666,100],

                        'who':['child', 'women', 'man', 'women'], 

                        'class':['First','First','First','Second'],

                       'adult_male': [True, False, True, False]})

newData
pd.concat([smallData, newData])
pd.concat([smallData, newData], join='inner')
df1 = pd.DataFrame({'employee_name':['Tasha','Linda', 'Olliver','Jack'],

                    'department':['Engineering', 'Accounting', 'HR', 'HR']})

df2 = pd.DataFrame({'employee_name':['Linda', 'Tasha', 'Jack', 'Olliver'],

                    'salary':[35000, 20500, 90000, 68000]})

print('----df1----\n',df1)

print('----df2----\n',df2)
df3 = pd.merge(df1,df2)

df3
df4 = pd.DataFrame({'department':['Engineering', 'Accounting', 'HR'],

                  'supervisor': ['Jonas', 'Martha', 'Martin']})

print('----df3----\n',df3)

print('----df4----\n',df4)

print('----merged----\n',pd.merge(df3, df4))
df5 = pd.DataFrame({'department':['Engineering', 'Engineering','Accounting',

                                  'Accounting', 'HR', 'HR'],

                  'skills': ['Coding', 'Soft skills', 'Math', 'Excel', 

                             'Organizing', 'Decision making']})

print('----df3----\n',df3)

print('----df5----\n',df5)

print('----merged----\n',pd.merge(df3, df5))
df2 = pd.DataFrame({'name':['Linda', 'Tasha', 'Jack', 'Olliver'],

                    'salary':[35000, 20500, 90000, 68000]})

print('--------df1---------\n',df1)

print('--------df2---------\n',df2)

print('-------merged--------\n',pd.merge(df1, df2, left_on='employee_name', right_on='name'))
df1=pd.DataFrame({'employee_name':['Tasha','Linda','Olliver','Jack'],  'department':['Engineering', 'Accounting', 'HR', 'HR']})

df2 = pd.DataFrame({'employee_name':['Linda', 'Mary'],'salary':[35000, 20500]})

print('--------df1---------\n',df1)

print('--------df2---------\n',df2)

print('\n-------merged--------\n',pd.merge(df1, df2))

print('-------left join--------\n',pd.merge(df1, df2, how='left'))

print('\n-------right join--------\n',pd.merge(df1,df2,how='right'))
print(df.groupby('sex'))

df.groupby('sex').sum()
data = df.groupby('sex')['survived'].sum()

print('% of male survivers',(data['male']/(data['male']+data['female']))*100)

print('% of male female',(data['female']/(data['male']+data['female']))*100)
df.groupby('sex')['survived'].aggregate(['sum', np.mean,'median'])
df.groupby('survived').filter(lambda x: x['fare'].std() > 50)
df.groupby('survived').transform(lambda x: x - x.mean())
def func(x):

    x['fare'] = x['fare'] / x['fare'].sum()

    return x

df.groupby('survived').apply(func)
df.groupby(['sex', 'pclass'])['survived'].aggregate('mean').unstack()
df.pivot_table('survived', index='sex', columns='pclass')
age = pd.cut(df['age'], [0, 18, 40, 80])

pivotTable = df.pivot_table('survived', ['sex', age], 'class')

pivotTable
pivotTable = pivotTable.unstack()

pivotTable
pivotTable.unstack(level=0)
pivotTable.stack()