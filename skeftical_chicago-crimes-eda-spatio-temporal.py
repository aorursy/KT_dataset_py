import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')
data = pd.read_csv("../input/Chicago_Crimes_2012_to_2017.csv")
data.drop(labels=['Case Number','ID', 'Block', 'Ward', 'Community Area', 'FBI Code', 'Year', 'Updated On',

                 'Latitude','Longitude', 'Beat'] , inplace=True, axis=1)
# convert dates to pandas datetime format

data.Date = pd.to_datetime(data.Date, format='%m/%d/%Y %I:%M:%S %p')

# setting the index to be the date will help us a lot later on

data.index = pd.DatetimeIndex(data.Date)
data.head(10)
data[['X Coordinate', 'Y Coordinate']] = data[['X Coordinate', 'Y Coordinate']].replace(0, np.nan)

data.dropna()

data.groupby('Primary Type').size().sort_values(ascending=False)
data.plot(kind='scatter',x='X Coordinate', y='Y Coordinate', c='District', cmap=plt.get_cmap('jet'))
plt.figure(figsize=(12,12))

sns.jointplot(x=data['X Coordinate'].values, y=data['Y Coordinate'].values, size=10, kind='hex')

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.show()
plt.figure(figsize=(12,12))

sns.lmplot(x='X Coordinate', y='Y Coordinate', size=10, hue='Primary Type', data=data, fit_reg=False)

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.show()
#Getting top k

topk = data.groupby(['District', 'Primary Type']).size().reset_index(name='counts').groupby('District').apply(lambda x: x.sort_values('counts',ascending=False).head(3))

print(topk)
#Plotting top-k per district

g =sns.factorplot("Primary Type", y='counts', col="District", col_wrap=4,

                   data=topk, kind='bar')

for ax in g.axes:

    plt.setp(ax.get_xticklabels(), visible=True, rotation=30, ha='right')



plt.subplots_adjust(hspace=0.4)
sdf = data.groupby(['District', 'Primary Type']).size().reset_index(name='counts')

idx = sdf.groupby(['District'])['counts'].transform(max) == sdf['counts']

sdf = sdf[idx]

other = data.groupby('District')[['X Coordinate', 'Y Coordinate']].mean()



sdf = sdf.set_index('District').join(other)

sdf = sdf.reset_index().sort_values("counts",ascending=False)

sns.lmplot(x='X Coordinate', y='Y Coordinate',size=10, hue='Primary Type', data=sdf,scatter_kws={"s": sdf['counts'].apply(lambda x: x/100.0)}, fit_reg=False)



for r in sdf.reset_index().as_matrix():



    district = "District {0}, Count : {1}".format(int(r[1]),int(r[3]))

    x = r[4]

    y = r[5]

    plt.annotate(

        district,

        xy=(x, y), xytext=(-20, 20),

        textcoords='offset points', ha='right', va='bottom',

        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),

        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
sdf.sort_values("counts",ascending=False)
g =sns.factorplot("Arrest", col="District", col_wrap=4, legend_out=True,

                   data=data, orient='h',

                    kind="count")

for ax in g.axes:

    plt.setp(ax.get_xticklabels(), visible=True)
g =sns.factorplot("Arrest", col="Primary Type", col_wrap=5, legend_out=True,

                   data=data, orient='h',

                    kind="count")

for ax in g.axes:

    plt.setp(ax.get_xticklabels(), visible=True)
data.resample('M').size().plot(legend=False)

plt.title('Number of crimes per month (2012 - 2017)')

plt.xlabel('Months')

plt.ylabel('Number of crimes')

plt.show()
crimes_per_district = data.pivot_table('Date', aggfunc=np.size, columns='District', index=data.index.date, fill_value=0)

crimes_per_district.index = pd.DatetimeIndex(crimes_per_district.index)

plo = crimes_per_district.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
#where does each crime occur more often

topk_loc_descr = data.groupby(['Primary Type', 'Location Description']).size().reset_index(name='counts').groupby('Primary Type').apply(lambda x: x.sort_values('counts',ascending=False).head(3))
pivotdf = topk_loc_descr.pivot(index='Primary Type', columns='Location Description', values='counts')

sns.heatmap(pivotdf)
#were most of crimes occured in residence and apartments causes of domestic violence ?

apartmentdf = data[data['Location Description']=='APARTMENT']

g =sns.factorplot("Domestic", col="Primary Type", col_wrap=4, legend_out=True,

                   data=apartmentdf, orient='h',

                    kind="count")

for ax in g.axes:

    plt.setp(ax.get_xticklabels(), visible=True)
#were most of crimes occured in residence and apartments causes of domestic violence ?

apartmentdf = data[data['Location Description']=='RESIDENCE']

g =sns.factorplot("Domestic", col="Primary Type", col_wrap=4, legend_out=True,

                   data=apartmentdf, orient='h',

                    kind="count")

for ax in g.axes:

    plt.setp(ax.get_xticklabels(), visible=True)