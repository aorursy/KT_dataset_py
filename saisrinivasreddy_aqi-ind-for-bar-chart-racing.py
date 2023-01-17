import pandas as pd
df = pd.read_csv('../input/air-quality-data-in-india/city_day.csv',parse_dates=['Date'])
df.head()
rdf = df[(df['Date'] > '2020-03-10')] 
rdf.head()
squashed = rdf[['City','Date','AQI']]
squashed.fillna(method='bfill',inplace=True)
squashed.info()
pivoted = squashed.pivot_table(index='City',columns='Date',values='AQI')
pivoted.columns = pivoted.columns.strftime('%b-%d')
pivoted.head()
pivoted.to_csv('AQI-IND.csv')
# grouped = squashed.groupby(['City','Date'])['AQI'].sum()
# gdf = grouped.to_frame()
# gdf