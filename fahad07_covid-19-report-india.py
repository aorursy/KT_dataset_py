import pandas as pd
df=pd.read_csv('../input/covid-19-dataset/datacovid.csv')
df
df.head(20)
df[df['countriesAndTerritories']=='India']
df['dateRep'].unique()
df[['month','dateRep','deaths']][df['countriesAndTerritories']=='India'].plot(kind='bar',x='dateRep',y=['month','deaths'],figsize=[20,6])
df[['dateRep','month','cases','deaths']].head(100).plot(kind='bar',x = 'dateRep', y= ['month','cases','deaths'],figsize = [40,6])
df[['dateRep','month','cases']].head(100).plot(kind='line',x = 'dateRep', y= ['month','cases'],figsize = [40,6])
df[['dateRep','month','cases']].head(100).plot(kind='scatter',x = 'month', y= ['cases'],figsize = [40,6])
df['month']
