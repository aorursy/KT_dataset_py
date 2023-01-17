import pandas as pd

df = pd.read_json("../input/nnDataSet.json").T
df.head()
df.Party.unique()
df.loc[df['Party']=='Republican','Party']='Rep'

df.loc[df['Party']=='Democrat','Party']='Dem'

df.loc[df['Party']=='I','Party']='Ind'
df.Position.unique()
df.loc[df['Position']=='House of Representatives','Position']='House'

df.loc[df['Position']=='US Senate','Position']='Senate'
df.info()
df['Party']=df['Party'].astype('category')

df['Position']=df['Position'].astype('category')

df['Vote']=df['Vote'].astype('category')

df['State']=df['State'].astype('category')

df['Contributions']=df['Contributions'].str.replace('$','')

df['Contributions']=df['Contributions'].str.replace(',','')

df['Contributions']=pd.to_numeric(df['Contributions'])
df.nlargest(10,'Contributions')
df.nsmallest(10,'Contributions')
gb = df.groupby('Vote').mean()

gb['Frequency'] = df['Vote'].value_counts();gb
df.groupby(['Vote','Party'])[['Contributions']].count().dropna()
demyes = df[(df['Vote']=='Yes') & (df['Party']=='Dem')]; demyes
repno = df[(df['Vote']=='No') & (df['Party']=='Rep')]; repno.sort_values('State')
sen = df[df['Position']=='Senate']

house = df[df['Position']=='House']
sen.head(3)
house.head(3)