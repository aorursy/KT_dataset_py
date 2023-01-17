import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

df = pd.read_csv('/kaggle/input/covid19-google-community-mobility-update-sept-2020/covid19_mobility_report_sept_2020/regional/2020_AU_Region_Mobility_Report.csv')
df.head()
# A new column 'month_name'

#df['month'] = pd.DatetimeIndex(df['date']).month_name()

df['month'] = pd.DatetimeIndex(df['date']).month
#Selecting some columns that want to be used for analysis



df1 = df[['sub_region_1','month','retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline' ]]
#Renaming columns to make it shorter

df1 = df1.rename({'retail_and_recreation_percent_change_from_baseline': 'retail_recreation', 'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy', 'parks_percent_change_from_baseline':'park', 'workplaces_percent_change_from_baseline':'workspace', 'residential_percent_change_from_baseline':'residential', 'transit_stations_percent_change_from_baseline':'transit_station'}, axis=1)
df1.shape
#Drop rows when sub_region is Nan

df2 = df1.dropna(subset=['sub_region_1'])
df2.head()
#Group by month and subregion

df3 = df2.groupby(['month','sub_region_1']).mean()
df3.head(50).sort_values(by="month")
qld = df2[df2['sub_region_1'] == 'Queensland']
qld = qld.groupby(['month','sub_region_1']).mean().reset_index()
qld
qld = qld.drop(columns=['sub_region_1'])

qld = qld.set_index('month')

qld
# create valid markers from mpl.markers

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if

item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

 

# valid_markers = mpl.markers.MarkerStyle.filled_markers

 

markers = np.random.choice(valid_markers, qld.shape[1], replace=False)

ax = qld.plot(kind='bar')

for i, line in enumerate(ax.get_lines()):

    line.set_marker(markers[i])



# adding legend

ax.legend(qld.columns, loc='best')

plt.ylabel('Percent Change from Baseline')

plt.xlabel('Month')

plt.title('Queensland Community Mobility ')

labels = ['Feb', 'Mar', 'Apr', 'May', 'June','July','Aug','Sept']

ax.set_xticklabels(labels)

plt.tight_layout()

plt.savefig('qld.png')

plt.show()
#Victoria



vic = df2[df2['sub_region_1'] == 'Victoria']

vic = vic.groupby(['month','sub_region_1']).mean().reset_index()

vic = vic.drop(columns=['sub_region_1'])

vic = vic.set_index('month')



# create valid markers from mpl.markers

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if

item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

 

# valid_markers = mpl.markers.MarkerStyle.filled_markers

 

markers = np.random.choice(valid_markers, vic.shape[1], replace=False)



ax = vic.plot(kind='bar')

for i, line in enumerate(ax.get_lines()):

    line.set_marker(markers[i])



# adding legend

ax.legend(vic.columns, loc='best')

plt.ylabel('Percent Change from Baseline')

plt.xlabel('Month')

plt.title('Victoria Community Mobility ')

labels = ['Feb', 'Mar', 'Apr', 'May', 'June','July','Aug','Sept']

ax.set_xticklabels(labels)

plt.tight_layout()

plt.savefig('vic.png')

plt.show()
#NSW



nsw = df2[df2['sub_region_1'] == 'New South Wales']

nsw = nsw.groupby(['month','sub_region_1']).mean().reset_index()

nsw = nsw.drop(columns=['sub_region_1'])

nsw = nsw.set_index('month')



# create valid markers from mpl.markers

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if

item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

 

# valid_markers = mpl.markers.MarkerStyle.filled_markers

 

markers = np.random.choice(valid_markers, nsw.shape[1], replace=False)



ax = nsw.plot(kind='bar')

for i, line in enumerate(ax.get_lines()):

    line.set_marker(markers[i])



# adding legend

ax.legend(nsw.columns, loc='best')

plt.ylabel('Percent Change from Baseline')

plt.xlabel('Month')

plt.title('NSW Community Mobility ')

labels = ['Feb', 'Mar', 'Apr', 'May', 'June','July','Aug','Sept']

ax.set_xticklabels(labels)

plt.tight_layout()

plt.savefig('nsw.png')

plt.show()