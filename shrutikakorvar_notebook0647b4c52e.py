import pandas as pd
df=pd.read_csv("../input/querying-data-frame/country_wise_latest.csv")

df.head()
df['Active'].head()
df['New cases'].max()
df['New cases'].mean()
df.loc[0:15,'Active':'New cases']
df.iloc[0:15, 0:10]
df[df['Active']==df['Active'].max()]['New cases']
df.sort_values(by='Active',ascending=True).head()
df.sort_values(by=['Active','New cases'],ascending=['True','False']).head()
df.groupby(by='Confirmed')['Active'].describe()
pd.crosstab(df['Active'],df['Confirmed'],normalize=True)