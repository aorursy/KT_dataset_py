# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
crime = pd.read_csv('../input/crime.csv', encoding='ISO-8859-1')

offense = pd.read_csv('../input/offense_codes.csv', encoding='ISO-8859-1')
crime.info()
crime.head()
print('Year 2015\'s record include months: ', set(crime[crime['YEAR'] == 2015]['MONTH']))

print('Year 2018\'s record include months: ', set(crime[crime['YEAR'] == 2018]['MONTH']))
crime['OFFENSE_DESCRIPTION'].unique()
# create a copy of the original dataframe, only perserving data from year 2016 and 2017 and wanted columns

crime_new = crime.query('YEAR == 2016 or YEAR == 2017')[['OFFENSE_DESCRIPTION', 'SHOOTING', 'OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'Lat', 'Long']]

# rename the columns 

crime_new.rename(columns={'OFFENSE_DESCRIPTION':'OFFENSE', 'OCCURRED_ON_DATE':'DATE'}, inplace=True)

# convert dates in string format to DatetimeIndex and set it as the new index for the dataframe

dates = pd.to_datetime(crime_new['DATE'])

crime_new.drop(axis=1, columns=['DATE'], inplace=True)

crime_new['DATE'] = dates

crime_new.set_index(keys=['DATE'], drop=True, inplace=True)

# sort the dataframe by index

crime_new.sort_index(axis=0, ascending=True, inplace=True)
print('There are {} types of crimes in this dataset'.format(len(set(crime['OFFENSE_DESCRIPTION']))))
crime_new.replace({'OFFENSE':[r'LARCENY [\w\W]+',

                            r'ASSAULT [\w\W]+',

                            r'AUTO THEFT [\w\W]+',

                            r'CHILD [\w\W]+',

                            r'ROBBERY [\w\W]+',

                            r'WEAPON [\w\W]+']},

                   {'OFFENSE':['LARCENY',

                            'ASSAULT',

                            'AUTO THEFT',

                            'CHILD',

                            'ROBBERY',

                            'WEAPON']}, 

                            regex=True,

                            inplace=True)



crime_df = crime_new[(crime_new['OFFENSE'] == 'LARCENY')|(crime_new['OFFENSE'] == 'ASSAULT')|

                    (crime_new['OFFENSE'] == 'AUTO THEFT')|(crime_new['OFFENSE'] == 'CHILD')|

                    (crime_new['OFFENSE'] == 'ROBBERY')|(crime_new['OFFENSE'] == 'WEAPON')]
# also, replace all NaN values in the SHOOTING column to string 'N'

crime_df.fillna({'SHOOTING': 'N'}, inplace=True)
Larceny = crime_df.query('OFFENSE == \'LARCENY\'').dropna()

Assault = crime_df.query('OFFENSE == \'ASSAULT\'').dropna()

Auto = crime_df.query('OFFENSE == \'AUTO THEFT\'').dropna()

Child = crime_df.query('OFFENSE == \'CHILD\'').dropna()

Robbery = crime_df.query('OFFENSE == \'ROBBERY\'').dropna()

Weapon = crime_df.query('OFFENSE == \'WEAPON\'').dropna()
from folium.plugins import HeatMap

#from folium.plugins import FastMarkerCluster, MarkerCluster



# map object's starting location

map = folium.Map(location=[42.306821, -71.060300],

                 zoom_start=12,

                 tiels='OpenStreetMap')



HeatMap(Auto[['Lat', 'Long']],

        min_opacity=0.8,

        radius=8,

        name='Auto Theft').add_to(map)



HeatMap(Larceny[['Lat', 'Long']],

        min_opacity=0.8,

        radius=8,

        show=False,

        name='Larceny').add_to(map)



HeatMap(Assault[['Lat', 'Long']],

        min_opacity=0.8,

        radius=8,

        show=False,

        name='Assault').add_to(map)



HeatMap(Child[['Lat', 'Long']],

        min_opacity=0.8,

        radius=8,

        show=False,

        name='Child').add_to(map)



HeatMap(Robbery[['Lat', 'Long']],

        min_opacity=0.8,

        radius=8,

        show=False,

        name='Robery').add_to(map)



HeatMap(Weapon[['Lat', 'Long']],

        min_opacity=0.8,

        radius=8,

        show=False,

        name='Weapon').add_to(map)





"""cluster = FastMarkerCluster(Auto_coord,

                            name='Auto Theft').add_to(map)"""





folium.LayerControl().add_to(map);
display(map)
# set the plot style

import seaborn as sns
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



grid1 = sns.catplot(x='MONTH', kind="count", palette="ch:.25", data=crime_df, aspect=2);

grid1.ax.set_title('Crimes by Month', fontsize=14)

grid1.ax.set_ylabel('NUMBER OF CRIMES')

grid1.ax.set_xticklabels(months);

grid1.ax.tick_params(axis='x', rotation=45)
palette = sns.cubehelix_palette(n_colors=7, start=2.5, rot=0, light=0.9)

order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



grid3 = sns.catplot(x='DAY_OF_WEEK', kind='count', data=crime_df, order=order, palette=palette, aspect=1.5)

grid3.ax.set_xlabel('DAY')

grid3.ax.set_ylabel('NUMBER OF CRIMES')

grid3.ax.set_title('Crimes by Day of a Week', fontsize=14);
palette = sns.cubehelix_palette(n_colors=24, start=3, rot=0, light=0.9)



grid2 = sns.catplot(x='HOUR', kind='count', data=crime_df, palette=palette, aspect=1.5)

grid2.ax.set_xlabel('HOUR')

grid2.ax.set_ylabel('NUMBER OF CRIMES')

grid2.ax.set_title('Crimes by Hour of a Day', fontsize=14);
Larceny_num = [len(Larceny[Larceny['MONTH'] == month]) for month in range(1,13)]

Assault_num = [len(Assault[Assault['MONTH'] == month]) for month in range(1,13)]

Auto_num = [len(Auto[Auto['MONTH'] == month]) for month in range(1,13)]

Robbery_num = [len(Robbery[Robbery['MONTH'] == month]) for month in range(1,13)]

Child_num = [len(Child[Child['MONTH'] == month]) for month in range(1,13)]

Weapon_num = [len(Weapon[Weapon['MONTH'] == month]) for month in range(1,13)]



crime_category = [Larceny_num, Assault_num, Auto_num, Robbery_num, Child_num, Weapon_num]

legends = ['Larceny', 'Assault', 'Auto Theft', 'Robbery', 'Child', 'Weapon']

months_numeric = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

plt.figure(figsize=(12,8))

for item in crime_category:

    sns.lineplot(x=months_numeric, y=item)

#plt.legend(legends, loc='lower center')

plt.legend(legends, bbox_to_anchor=(1.02, 1), ncol=1, loc=2)

plt.xticks(ticks=months_numeric, labels=months);

plt.grid(b=True, axis='x', alpha=0.5)

plt.title('Crimes by Month by Category', fontsize=14)

plt.xlabel('MONTH')

plt.ylabel('NUMBER OF CRIMES');
# extract the month and day values from the datatime index

month_day = crime_df.index.strftime('%m-%d')

crime_df['MONTH_DAY'] = month_day
# create a pivot table counting the number of offenses occured on each day of month

pivot = pd.pivot_table(data=crime_df, index='MONTH_DAY', aggfunc=['count'])

pivot.columns = pivot.columns.droplevel(level=0)
plt.figure(figsize=(20,8))

sns.lineplot(x=pivot.index, y='OFFENSE', data=pivot)

#plt.autoscale(enable=True, axis='x', tight=True)

plt.xticks(ticks=[16,46,77,107,138,168,199,229,260,290,321,351],

           labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.xlabel('MONTH')

plt.title('Number of Crimes by Days', fontsize=14)



# label the important days in a year

important_dates = [0, 76, 84, 128, 200, 248, 315, 358]

day_names = ['New Year\'s Day', 'St.Patrick\'s Day', 'Good Friday', 'Mother\'s Day', 'Father\'s Day', 'Labor Day',

             'Veterans Day', 'Christmas Eve']

offense_num = [pivot.iloc[date].OFFENSE for date in important_dates]



for date in important_dates:

    plt.axvline(x=date, ymin=0, ymax=1, dashes=[5,2,1,2], color='grey', alpha=0.75)



plt.text(x=important_dates[0], y=155, s='{}\n{}'.format(day_names[0], offense_num[0]))

plt.text(x=important_dates[1], y=150, s='{}\n{}'.format(day_names[1], offense_num[1]))

plt.text(x=important_dates[2], y=65, s='{}\n{}'.format(day_names[2], offense_num[2]))

plt.text(x=important_dates[3], y=155, s='{}\n{}'.format(day_names[3], offense_num[3]))

plt.text(x=important_dates[4], y=70, s='{}\n{}'.format(day_names[4], offense_num[4]))

plt.text(x=important_dates[5], y=65, s='{}\n{}'.format(day_names[5], offense_num[5]))

plt.text(x=important_dates[6], y=65, s='{}\n{}'.format(day_names[6], offense_num[6]))

plt.text(x=important_dates[7]-10, y=155, s='{}\n{}'.format(day_names[7], offense_num[7]));