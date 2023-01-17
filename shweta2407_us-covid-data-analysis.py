import pandas as pd
data= pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv', engine='python')
data.head()
#total number of cases due to covid
data['cases'].aggregate(['sum'])
#total number of cases due to covid
data['deaths'].aggregate(['sum'])
#drop fips column
data = data.drop(['fips'], axis=1) 

#group the data by state and find the total sum of deaths and cases occured in that state 
df = data.groupby('state').sum()
df[:5]
#state with maximum number of cases
df[df['cases']== df['cases'].max()]
#state with minimum number of cases
df[df['cases']== df['cases'].min()]
#similarly do with the country missplelled as 'county' here
d = data.groupby('county').sum()
d[d['cases'] == d['cases'].max()]
#plot the graph to see the top 10 cities with maximum number of deaths
df.sort_values('deaths', ascending=False)[:10].plot.bar(stacked=True,figsize=(20,10), rot=0)
#plot the graph to see the top 10 cities with maximum number of deaths
df.sort_values('cases', ascending=False)[:10].plot.bar(figsize=(20,10), rot=0)
#plot the line graph to see the surge in covid cases after a particular date (21-03-2020)
data.groupby('date').sum().sort_values('deaths', ascending=True).plot.line(figsize=(20,10))
#on which date (24 june 2020) maximum death occurred

data.groupby('date').sum().sort_values('deaths', ascending=False)[:10]
#in which state on which date max death occurred 

data.groupby(['state', 'date']).sum().sort_values('deaths', ascending=False)
