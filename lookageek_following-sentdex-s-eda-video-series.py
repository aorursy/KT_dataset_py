import pandas as pd

%matplotlib inline

df = pd.read_csv('../input/avocado.csv')
# Let's glance at what we have here

df.head()
df['Date'] = pd.to_datetime(df['Date'])
# Picking up only albany data by using selector on dataframe

albany_df = df.copy()[df['region'] == 'Albany']
albany_df = albany_df.set_index("Date")
albany_df['AveragePrice'].plot()
albany_df['AveragePrice'].rolling(25).mean().plot()
albany_df.index
albany_df.sort_index(inplace=True)

albany_df['AveragePrice'].rolling(25).mean().plot()
albany_df['price25ma'] = albany_df['AveragePrice'].rolling(25).mean()
organic_df = df.copy()[df['type']=='organic']

organic_df['Date'] = pd.to_datetime(organic_df['Date'])

df.sort_values(by='Date', ascending=True, inplace=True)
# Now transform the organic_df to the structure mentioned above

graph_df = pd.DataFrame()



for region in organic_df['region'].unique():

    region_df = organic_df.copy()[organic_df['region']==region]

    region_df.set_index('Date', inplace=True)

    region_df.sort_index(inplace=True)

    region_df[f'{region}_price25ma'] = region_df['AveragePrice'].rolling(25).mean()

    

    if graph_df.empty:

        graph_df = region_df[[f'{region}_price25ma']]

    else:

        graph_df = graph_df.join(region_df[f'{region}_price25ma'])

    
graph_df.tail()
# Plotting! Making plot a bit bigger to see more clearly the graph, turning off the legend, dropping na, rolling

# average for first 25 rows will be NaN - to make graph look lit!

graph_df.dropna().plot(figsize=(14, 10), legend=False)