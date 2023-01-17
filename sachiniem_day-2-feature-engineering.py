import pandas as pd

length=[10,18,12,26,9,30]

breadth=[11,23,10,27,13,31]

typ=['Medium','Large','Medium','Large','Medium','Large']

df=pd.DataFrame({'Length':length,'Breadth':breadth,'Type':typ})

df.head()
df['Area']=df['Length']*df['Breadth']

df
city=['A','B','C','D','E','F']

roll=[12,14,13,16,17,19]

df2=pd.DataFrame({'City':city,'Roll':roll})

df2