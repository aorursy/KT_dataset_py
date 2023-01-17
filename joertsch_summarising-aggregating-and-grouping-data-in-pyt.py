# Import modules

import pandas as pd

import dateutil



data = pd.read_csv('../input/phone_data.csv')



# Print the first few rows

data.head()
df = pd.read_csv('../input/phone_data.csv')

print(df.isnull().sum())
type(data['date'])
#reading dates

data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True)

data.info()
# How many rows the dataset

data['item'].count()
# What was the longest phone call / data entry?

seconds = data['duration'].max()

# in seconds

minutes = seconds/60

hours = minutes/60

hours

# How many seconds of phone calls are recorded in total (hours)?

data['duration'][data['item'] == 'call'].sum()/3600
# How many entries are there for each month?

entries_month = data.groupby('month')['item'].count()

#entries_month.plot(x='month', y='item')

entries_month 
# How many entries are there for each month?

data['month'].value_counts()
# Number of non-null unique network entries

data['network'].nunique()

data.network.unique()
#Keys der Dictionary nach Month

data.groupby(['month']).groups.keys()
#Einträge im Monat November

len(data.groupby(['month']).groups['2014-11'])
# Get the first entry for each month

data.groupby('month').first()

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.first.html
# Get the sum of the durations per month in hours

data.groupby('month')['duration'].sum()/3600
# Get the number of dates / entries in each month

data.groupby(['month'])['date'].count()
# What is the sum of durations, for calls only, to each network in hours

networkhours =data[data['item'] == 'call'].groupby('network')['duration'].sum()/3600

type(networkhours)

networkhours.plot.bar(x='network')
data
data.groupby('network')['network'].count()
import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="ticks")



# Initialize the figure with a logarithmic y axis

f, ax = plt.subplots(figsize=(7, 6))

ax.set_yscale("log")



# Plot the orbital period with horizontal boxes

sns.boxplot(x="network", y="duration", data=data[['network','duration']], palette="vlag")



# Tweak the visual presentation

ax.yaxis.grid(True)

ax.set(ylabel="duration")

sns.despine(trim=True, left=True)
cond_plot = sns.FacetGrid(data=data, col='network', hue='network_type', col_wrap=4)

cond_plot.map(sns.barplot, 'network', 'duration');
# How many calls, sms, and data entries are in each month?

data.item.unique()

data.groupby(['month', 'item'])['date'].count()
# How many calls, texts, and data are sent per month, split by network_type?

data.groupby(['month','network_type'])['date'].count()
# produces Pandas Series

data.groupby('month')['duration'].sum() 
# Produces Pandas DataFrame

data.groupby('month')[['duration']].sum()
data.groupby('month', as_index=False).agg({"duration": "sum"})
# Group the data frame by month and item and extract a number of stats from each group

data.groupby(

    ['month', 'item']

).agg(

    {

            'duration':sum,    # Sum duration per group

            'network_type': "count",  # get the count of networks

            'date': 'first'  # get the first date per group

    }

)