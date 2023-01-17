# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# trying to improve plot appearance
import seaborn as sns
sns.set()
#COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
#COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
#covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
confirmed.columns
#confirmed.dtypes
#combined country and state columns
confirmed['state_length'] = confirmed['Province/State'].str.len().fillna(0)
confirmed['Country_State'] = confirmed['Country/Region'] + np.where(confirmed['state_length']>0, '_'+confirmed['Province/State'], '')
confirmed.head(5)
# time series data needs to be melted to then allow dates to be read in correctly
df_cols_index = confirmed.columns
df_cols_list = df_cols_index.tolist()
#remove non date columns - this allows dynamic update and melting of data
not_dates = ['Province/State','Country/Region','Lat','Long','state_length','Country_State']
date_list = []
for dates in df_cols_list:
    if dates not in not_dates :
        date_list.append(dates)

melted = pd.melt(confirmed, id_vars=['Country/Region','Country_State'], value_vars=date_list)
melted['Date'] = pd.to_datetime(melted['variable'], infer_datetime_format=True)
#this allows for growth to be shown against the days since the first discovery in that country/state
#earliest_date[FirstObservedDate] = all_discovered.groupby('Country_State')['Date'].min()

#filter out any 0's before any discovery
all_discovered = melted[melted['value']>0]
earliest_date = all_discovered.groupby(["Country_State"])[['Date']].min().reset_index()
earliest_date.rename(columns={'Date':'EarliestConfirmedDate'}, inplace=True)
#join with main discovered dataset
data_withearliest = all_discovered.merge(earliest_date,how = 'inner',  left_on=['Country_State'], right_on=['Country_State'], suffixes = ['_l','_r'])
data_withearliest['DaysFromStart'] = (data_withearliest['Date'] - data_withearliest['EarliestConfirmedDate']).dt.days
data_withearliest.head()
# plotting all countries is impossible, so wanted to rank coutries based on Total discovered
max_discovered = all_discovered.groupby(['Country/Region','Country_State'])[['value']].max().reset_index()
sum_discovered = all_discovered.groupby(['Country/Region','Country_State'])[['value']].sum().reset_index()
sd = sum_discovered.sort_values('value',ascending=False)
sd.head(25)
max_discovered['Country_Rank'] = max_discovered['value'].rank()
md = max_discovered[['Country_Rank','Country_State']]
# add rank data to main dataset
data_withrank = data_withearliest.merge(md, how = 'inner',  left_on=['Country_State'], right_on=['Country_State'], suffixes = ['_l','_r'])

# I have used two approaches to subplots - this approach uses legends to show the country
# I had no control over where the legends were placed

# just China
datatoplot = data_withrank[(data_withrank['Country_Rank']>0) & (data_withrank['Country/Region'] == 'China') ]
pivoted = datatoplot.pivot(index='DaysFromStart', columns='Country_State', values='value')
#pivoted.plot(figsize = (16,16), layout=(10,6), subplots=True)
#lines = pivoted.plot.line(x='Date', y='value').legend(title='Country_State')
pivoted.plot(figsize = (16,16),layout=(10,6), subplots=True, logy = True);
# not china
datatoplot = data_withrank[(data_withrank['Country_Rank']>220) & (data_withrank['Country/Region'] != 'China') ]
pivoted = datatoplot.pivot(index='DaysFromStart', columns='Country_State', values='value')
#pivoted.plot(figsize = (16,16), layout=(10,6), subplots=True)
#lines = pivoted.plot.line(x='Date', y='value').legend(title='Country_State')
pivoted.plot(figsize = (16,16),layout=(10,6), subplots=True, logy = True, legend='reverse',sharey=True );
# more control over sub plots
import matplotlib.pyplot as plt
#pivoted
# example found to make sure sub plot were working
def f(t):
    s1 = np.cos(2*np.pi*t)
    e1 = np.exp(-t)
    return s1 * e1

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)


fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(t1, f(t1), 'o', t2, f(t2), '-')
axs[0].set_title('subplot 1')
axs[0].set_xlabel('distance (m)')
axs[0].set_ylabel('Damped oscillation')
fig.suptitle('This is a somewhat long figure title', fontsize=16)

axs[1].plot(t3, np.cos(2*np.pi*t3), '--')
axs[1].set_xlabel('time (s)')
axs[1].set_title('subplot 2')
axs[1].set_ylabel('Undamped')

plt.show()
# looking at how to split up the countries
#fig, ax = plt.subplots()
#fig = plt.figure()
countryranksbuckets = data_withrank.groupby(['Country/Region'])[['Country_Rank']].max().reset_index()
BucketSize = 20
countryranksbuckets['Bucket'] = countryranksbuckets['Country_Rank']/BucketSize

#countryranksbuckets['Rounded_Bucket'] = countryranksbuckets.round({countryranksbuckets['Bucket']:0})
newdf = countryranksbuckets.round({"Bucket":0})
newdf['RankMin'] = (newdf['Bucket']*BucketSize) - BucketSize
newdf['RankMax'] = (newdf['Bucket']*BucketSize)
#newdf.groupby(['Country/Region'])[['Country_Rank']].max().reset_index()
newdf1 = newdf.groupby(['Bucket','RankMin','RankMax']).count().reset_index()
newdf1['CumSum'] = newdf1['Country_Rank'].cumsum(axis = 0) 
newdf1
datatoplot = data_withrank[(data_withrank['Country_Rank']>200) & (data_withrank['Country/Region'] != 'China') ]
#figuredata = datatoplot[datatoplot['Country_State'] == 'China_Hubei']
#figuredata
# create a dynamic sqaure matrix of sub plots - maybe set it to a width of 6 
import math
countries = datatoplot.Country_State.unique().tolist();
howmanycountries = len(countries)
squaredimensions = int(round(math.sqrt(howmanycountries),0));

#fig, axs = plt.subplots(squaredimensions, squaredimensions, constrained_layout=True)
# this loops through the dataframe and sets the parameters for each sub plot
# it allows the country/state to be set as the figure title
fig, axs = plt.subplots(squaredimensions, squaredimensions, constrained_layout=True, figsize = (14,14))
i = 0
j = 0
for c in countries:
    try:
        figuredata = pd.DataFrame()
        figuredata = datatoplot[datatoplot['Country_State'] == c]
        xdata = figuredata['DaysFromStart'].tolist();
        ydata = figuredata['value'].tolist();
        axs[i,j].axis('off')
        axs[i,j].plot(xdata, ydata)
        axs[i,j].set_title(c)
        i = i + 1
        if i == squaredimensions:
            i = 0
            j = j + 1
        if (i == 2 and j == 2):
            pass
            #print(figuredata)
    except:
        pass

fig.savefig('foo.png')
    
#ax = fig.add_subplot(616)

#index='DaysFromStart', columns='Country_State', values='value'
#datatoplot.groupby('Country_State').plot(x='DaysFromStart', y='value')
#df.groupby('country').plot(x='year', y='unemployment', ax=ax, legend=False)
#datatoplot = data_withrank[(data_withrank['Country_Rank']>290) & (data_withrank['Country/Region'] == 'China') ]
#countries = datatoplot.Country_State.unique().tolist();
#countries
#countries_list = countries.tolist
#countries_list
#ax.set_title = countries
#datatoplot.groupby('Country_State').plot(x='DaysFromStart', y='value', legend=True,logy=True, subplots=True,figsize = (10,10),layout=(6,6) );