import pandas as pd

df = pd.DataFrame({"col1":[1,3,11,2], "col2":[9,23,0,2]})

print(type(df))

df
df.iloc[:2,:2]
df['col1']
#df.sum(axis=1)

df.var(axis=0)

#df.min()

#df.max()

#df.std()
df = pd.DataFrame({'col1':[1,1,0,0,2,2,2], 'col2':[1,2,3,4,5,6,7]}) 

df
grp=df.groupby('col1')

# selección de los que corresponden a col1=1

grp.get_group(1)
grp.get_group(0)
grp.get_group(2)
#grp.get_group(0)

grp.sum() 

grp.mean()
grp.mean()
grp.count()
df['sum_col']=df.eval('col1+col2') 

df
grp=df.groupby(['sum_col','col1']) 
res=grp.sum()

res
df['sin_col2']=df.eval('sin(col2)')

df
grp=df.groupby(['sum_col','col1'])

res=grp.sum()

res