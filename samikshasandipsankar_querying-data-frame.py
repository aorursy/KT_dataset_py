import pandas as pd
df=pd.read_csv(r"../input/python-pandas-data-frame/prog_book.csv")

df.head()
df[df['Type'] == 'Kindle Edition']
df.query("Type == 'Kindle Edition'")
df.query("Type == 'Kindle Edition' & Rating >4")
df['Book_title'].head()
df['Number_Of_Pages'].min()
df['Number_Of_Pages'].max()
df['Number_Of_Pages'].mean()
df['Number_Of_Pages'].value_counts()
df.loc[0:4,'Book_title']
df.loc[0:7,'Book_title': 'Number_Of_Pages']
df.iloc[0:6,0:7]
df[df['Rating']==df['Rating'].max()]['Price']
df[df['Rating']==df['Rating'].min()]['Price']
df.sort_values(by=['Book_title','Type'],ascending=[True,False]).head()
df.sort_values(by=['Book_title','Type'],ascending=[True,False])
df.sort_values(by='Rating',ascending=True).head()
df.head()
d={'3,829': 5460,'1,406': 2890}

df['Reviews']=df['Reviews'].map(d)

df.head()
o={5460:'3,829',2890:'1,406'}

df=df.replace({'Reviews':o})

df.head()
df.groupby(by='Type')['Price'].describe()
pd.crosstab(df['Reviews'],df['Price'],normalize=True)
df.pivot_table(['Rating','Price'],['Type'],aggfunc='mean')