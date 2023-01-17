# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import calendar

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from mpl_toolkits.basemap import Basemap

from IPython.core.display import display, HTML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Mass Shootings Dataset.csv', encoding = 'ISO-8859-1', parse_dates=['Date'])

df.drop(['S#'], axis=1, inplace=True)

df.head()
df['Gender'].fillna('Male', inplace=True)
print(df['Gender'].unique())
df.Gender.replace(['M', 'F', 'M/F'], ['Male', 'Female', 'Male/Female'], inplace=True)
print(df['Gender'].unique())
df.groupby('Gender').count()
df['Year'] = df['Date'].dt.year
fatalities_year = df[['Fatalities', 'Year']].groupby('Year').sum()



fatalities_year.plot.bar(figsize=(12,6), color='red')

plt.ylabel('Fatalities', fontsize=12)

plt.title('Number of Fatalities per Year', fontsize=18)
fatalities_year
fatalities_year[fatalities_year['Fatalities'] > 100]
injured_year = df[['Injured', 'Year']].groupby('Year').sum()



injured_year.plot.bar(figsize=(12,6))

plt.ylabel('Injured', fontsize=12)

plt.title('Number of Injured per Year', fontsize=18)
injured_year
injured_year[injured_year['Injured'] > 100]
tot_victims = df[['Year', 'Injured', 'Fatalities']].groupby('Year').sum()



tot_victims.plot.bar(figsize=(12,6))

plt.ylabel('Number of Victims', fontsize=12)

plt.title('Number of Fatalities vs Injuries per Year', fontsize=18)
df[['Year','Fatalities', 'Injured', 'Total victims']].groupby('Year').sum()
print('Total Fatalities: ' + str(df['Fatalities'].sum()))
print('Total Injured: ' + str(df['Injured'].sum()))
print('Total Number of Victims: ' + str(df['Total victims'].sum()))
year_count = df['Year'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(year_count.index, year_count.values, alpha=0.8, color=color[2])

plt.xticks(rotation='vertical')

plt.xlabel('Year of Shooting', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Number of Attacks per Year', fontsize=18)

plt.show()
# U.S. center lat and long

center_lat = 39.8283

center_lon = -98.5795



df_positions = df[['Latitude', 'Longitude', 'Total victims']].dropna()
plt.figure(figsize=(16,8))



latitudes = np.array(df_positions['Latitude'])

longitudes = np.array(df_positions['Longitude'])



lons, lats = np.meshgrid(longitudes,latitudes)



m = Basemap(projection='mill',llcrnrlat=20,urcrnrlat=50,\

                llcrnrlon=-130,urcrnrlon=-60,resolution='c')

m.drawcoastlines()

m.drawcountries()

m.drawstates()

m.fillcontinents(color='#04BAE3', lake_color='#FFFFFF')

m.drawmapboundary(fill_color='#FFFFFF')



x, y = m(longitudes, latitudes)



m.plot(x, y, 'ro')



plt.title("Mass Shooting Attacks on US Map")
gender = df['Gender'].value_counts()

gender
gender.sort_values().plot(kind='bar', figsize=(12,6), fontsize=12).set_title(

    'Gender Distribution of Mass Shootings')
gender_prop = pd.Series()



for key, value in gender.iteritems():

    gender_prop[key] =  gender[key] / len(df) * 100

    

gender_prop
df[['Gender', 'Total victims']].groupby('Gender').sum()
df['Race'].value_counts()
df.loc[['white' in str(x).lower().split() for x in df['Race']], 'Race']= 'white'

df.loc[['black' in str(x).lower().split() for x in df['Race']], 'Race']= 'black'

df.loc[['asian' in str(x).lower().split() for x in df['Race']], 'Race']= 'asian'

df.loc[['native' in str(x).lower().split() for x in df['Race']], 'Race']= 'native american'



race_counts = df['Race'].value_counts()

race_counts
race_counts.sort_values().plot(kind='bar', figsize=(12,6), fontsize=12).set_title('Race distribution of mass shootings')
race_prop = pd.Series()



for key, value in race_counts.iteritems():

    race_prop[key] = (race_counts[key] / len(df)) * 100

    

race_prop
df['Month'] = df['Date'].dt.month

df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
month = df['Month'].value_counts()

month.sort_index().plot.bar(figsize=(12,6), alpha=0.8, color=color[1])

plt.ylabel('Number of Shooting', fontsize=12)

plt.title('Number of Shootings per Month', fontsize=18)
month_df = df[['Month', 'Total victims']].groupby('Month').sum()

month_df.plot.bar(figsize=(12,6), alpha=0.8, color=color[3])

plt.ylabel('Total Number of Victims', fontsize=12)

plt.title('Number of Victims per Month', fontsize=18)
df.loc[['unknown' in str(x).lower().split() for x in df['Mental Health Issues']], 'Mental Health Issues']= 'Unknown'

df.loc[['unclear' in str(x).lower().split() for x in df['Mental Health Issues']], 'Mental Health Issues']= 'Unclear'



mental_issues = df['Mental Health Issues'].value_counts()



print(mental_issues)
df.set_index('Date', inplace=True)

df.sort_index(inplace=True)
ax = df[df['Mental Health Issues'] == 'Yes']['Total victims'].plot(style='o', label='Yes', figsize=(14,10))

df[df['Mental Health Issues'] == 'No']['Total victims'].plot(style='o', label='No', ax=ax)

df[df['Mental Health Issues'] == 'Unknown']['Total victims'].plot(style='o', label='Unknown', ax=ax)

plt.title('Mental Health Issues')

ax.legend()