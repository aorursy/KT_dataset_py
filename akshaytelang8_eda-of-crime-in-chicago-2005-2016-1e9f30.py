import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn')



# we will skip 2001 - 2005 due to bad quality



crimes1 = pd.read_csv('../input/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

crimes2 = pd.read_csv('../input/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

crimes3 = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)

crimes = pd.concat([crimes1, crimes2, crimes3], ignore_index=False, axis=0)



del crimes1

del crimes2

del crimes3



print('Dataset ready..')



print('Dataset Shape before drop_duplicate : ', crimes.shape)

crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)

print('Dataset Shape after drop_duplicate: ', crimes.shape)
crimes.drop(['Unnamed: 0', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District'], inplace=True, axis=1)
#Let's have a look at the first 3 records and see if we see what we expect

crimes.head(3)
# convert dates to pandas datetime format

crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')

# setting the index to be the date will help us a lot later on

crimes.index = pd.DatetimeIndex(crimes.Date)
# of records X # of features

crimes.shape
crimes.info()
loc_to_change  = list(crimes['Location Description'].value_counts()[20:].index)

desc_to_change = list(crimes['Description'].value_counts()[20:].index)

#type_to_change = list(crimes['Primary Type'].value_counts()[20:].index)



crimes.loc[crimes['Location Description'].isin(loc_to_change) , crimes.columns=='Location Description'] = 'OTHER'

crimes.loc[crimes['Description'].isin(desc_to_change) , crimes.columns=='Description'] = 'OTHER'

#crimes.loc[crimes['Primary Type'].isin(type_to_change) , crimes.columns=='Primary Type'] = 'OTHER'
# we convert those 3 columns into 'Categorical' types -- works like 'factor' in R

crimes['Primary Type']         = pd.Categorical(crimes['Primary Type'])

crimes['Location Description'] = pd.Categorical(crimes['Location Description'])

crimes['Description']          = pd.Categorical(crimes['Description'])
plt.figure(figsize=(11,5))

crimes.resample('M').size().plot(legend=False)

plt.title('Number of crimes per month (2005 - 2016)')

plt.xlabel('Months')

plt.ylabel('Number of crimes')

plt.show()
plt.figure(figsize=(11,4))

crimes.resample('D').size().rolling(365).sum().plot()

plt.title('Rolling sum of all crimes from 2005 - 2016')

plt.ylabel('Number of crimes')

plt.xlabel('Days')

plt.show()
crimes_count_date = crimes.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=crimes.index.date, fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)

plo = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']

crimes.groupby([crimes.index.dayofweek]).size().plot(kind='barh')

plt.ylabel('Days of the week')

plt.yticks(np.arange(7), days)

plt.xlabel('Number of crimes')

plt.title('Number of crimes by day of the week')

plt.show()
crimes.groupby([crimes.index.month]).size().plot(kind='barh')

plt.ylabel('Months of the year')

plt.xlabel('Number of crimes')

plt.title('Number of crimes by month of the year')

plt.show()
plt.figure(figsize=(8,10))

crimes.groupby([crimes['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')

plt.title('Number of crimes by type')

plt.ylabel('Crime Type')

plt.xlabel('Number of crimes')

plt.show()
plt.figure(figsize=(8,10))

crimes.groupby([crimes['Location Description']]).size().sort_values(ascending=True).plot(kind='barh')

plt.title('Number of crimes by Location')

plt.ylabel('Crime Location')

plt.xlabel('Number of crimes')

plt.show()
hour_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.hour, aggfunc=np.size).fillna(0)

hour_by_type     = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.hour, aggfunc=np.size).fillna(0)

hour_by_week     = crimes.pivot_table(values='ID', index=crimes.index.hour, columns=crimes.index.weekday_name, aggfunc=np.size).fillna(0)

hour_by_week     = hour_by_week[days].T # just reorder columns according to the the order of days

dayofweek_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)

dayofweek_by_type = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)

location_by_type  = crimes.pivot_table(values='ID', index='Location Description', columns='Primary Type', aggfunc=np.size).fillna(0)
from sklearn.cluster import AgglomerativeClustering as AC



def scale_df(df,axis=0):

    '''

    A utility function to scale numerical values (z-scale) to have a mean of zero

    and a unit variance.

    '''

    return (df - df.mean(axis=axis)) / df.std(axis=axis)



def plot_hmap(df, ix=None, cmap='bwr'):

    '''

    A function to plot heatmaps that show temporal patterns

    '''

    if ix is None:

        ix = np.arange(df.shape[0])

    plt.imshow(df.iloc[ix,:], cmap=cmap)

    plt.colorbar(fraction=0.03)

    plt.yticks(np.arange(df.shape[0]), df.index[ix])

    plt.xticks(np.arange(df.shape[1]))

    plt.grid(False)

    plt.show()

    

def scale_and_plot(df, ix = None):

    '''

    A wrapper function to calculate the scaled values within each row of df and plot_hmap

    '''

    df_marginal_scaled = scale_df(df.T).T

    if ix is None:

        ix = AC(4).fit(df_marginal_scaled).labels_.argsort() # a trick to make better heatmaps

    cap = np.min([np.max(df_marginal_scaled.as_matrix()), np.abs(np.min(df_marginal_scaled.as_matrix()))])

    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)

    plot_hmap(df_marginal_scaled, ix=ix)

    

def normalize(df):

    result = df.copy()

    for feature_name in df.columns:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result
plt.figure(figsize=(15,12))

scale_and_plot(hour_by_type)
plt.figure(figsize=(15,7))

scale_and_plot(hour_by_location)
plt.figure(figsize=(12,4))

scale_and_plot(hour_by_week, ix=np.arange(7))
plt.figure(figsize=(17,17))

scale_and_plot(dayofweek_by_type)
plt.figure(figsize=(15,12))

scale_and_plot(dayofweek_by_location)
df = normalize(location_by_type)

ix = AC(3).fit(df.T).labels_.argsort() # a trick to make better heatmaps

plt.figure(figsize=(17,13))

plt.imshow(df.T.iloc[ix,:], cmap='Reds')

plt.colorbar(fraction=0.03)

plt.xticks(np.arange(df.shape[0]), df.index, rotation='vertical')

plt.yticks(np.arange(df.shape[1]), df.columns)

plt.title('Normalized location frequency for each crime')

plt.grid(False)

plt.show()