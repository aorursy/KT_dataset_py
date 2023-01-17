import pandas as pd



df=pd.read_csv("../input/groceries-dataset/Groceries_dataset.csv")



df.head()
df.pivot_table(['Member_number','Date'],['Date'],aggfunc='mean')
pd.crosstab(df['Member_number'],df['Date'],normalize=True)
df.groupby(by='Date')['Member_number'].describe()
o={5460:'3,829',2890:'1,406'}

df=df.replace({'itemDescription':o})

df.head()
d={'3,829': 5460,'1,406': 2890}

df['itemDescription']=df['itemDescription'].map(d)

df.head()
df.head()
df.sort_values(by='Member_number',ascending=True).head()
df.sort_values(by=['itemDescription','Member_number'],ascending=[True,False])
df.sort_values(by=['itemDescription','Member_number'],ascending=[True,False]).head(),
df[df['Member_number']==df['Member_number'].min()]['Date']
df[df['Member_number']==df['Member_number'].max()]['Date']
df.iloc[0:6,0:7]
df.loc[0:7,'itemDescription': 'Member_number']
df.loc[0:4,'itemDescription']
df['Member_number'].value_counts()
df['Member_number'].mean()
df['Member_number'].max()
df['Member_number'].min()
df['Date'].head()
df.query("itemDescription == 'whole milk' & Member_number >4")
df.query("itemDescription == 'whole milk'")
df[df['itemDescription'] == 'whole milk']