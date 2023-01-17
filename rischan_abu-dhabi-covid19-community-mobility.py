import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

df = pd.read_csv('/kaggle/input/covid19-google-community-mobility-update-sept-2020/covid19_mobility_report_sept_2020/regional/2020_AE_Region_Mobility_Report.csv')
df.head()
df['date'] = pd.to_datetime(df['date'])

df['week_number'] = df['date'].dt.week
df.head()
# A new column 'month_name'

#df['month'] = pd.DatetimeIndex(df['date']).month_name()

df['month'] = pd.DatetimeIndex(df['date']).month
df = df[~df['month'].isin([7,8,9])]
df.head()
#Selecting some columns that want to be used for analysis



df1 = df[['sub_region_1', 'week_number','retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline']]
#Renaming columns to make it shorter

df1 = df1.rename({'retail_and_recreation_percent_change_from_baseline': 'retail_recreation', 'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy', 'parks_percent_change_from_baseline':'park', 'workplaces_percent_change_from_baseline':'workspace', 'residential_percent_change_from_baseline':'residential', 'transit_stations_percent_change_from_baseline':'transit_station'}, axis=1)
df1.shape
#Drop rows when sub_region is Nan

df2 = df1.dropna(subset=['sub_region_1'])
df2.head()
#Group by month and subregion

df3 = df2.groupby(['week_number','sub_region_1']).mean()
df3.head(50).sort_values(by="week_number")
ab = df2[df2['sub_region_1'] == 'Abu Dhabi']
ab = ab.groupby(['week_number','sub_region_1']).mean().reset_index()
ab
ab = ab.drop(columns=['sub_region_1'])

ab = ab.set_index('week_number')

ab
# create valid markers from mpl.markers

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if

item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

 

# valid_markers = mpl.markers.MarkerStyle.filled_markers

 

markers = np.random.choice(valid_markers, ab.shape[1], replace=False)

ax = ab.plot(kind='bar')

for i, line in enumerate(ax.get_lines()):

    line.set_marker(markers[i])



# adding legend

ax.legend(ab.columns, loc='best')

plt.ylabel('Percent Change from Baseline')

plt.xlabel('Week number of the year 2020')

plt.title('Abu Dhabi Community Mobility ')

#labels = ['Feb', 'Mar', 'Apr', 'May', 'June','July','Aug','Sept']

#ax.set_xticklabels(labels)

plt.tight_layout()

fig = mpl.pyplot.gcf()

fig.set_size_inches(18.5, 10.5)

plt.savefig('ab.png')

plt.show()
ab.to_csv('abu_dhabi.csv', index=True)