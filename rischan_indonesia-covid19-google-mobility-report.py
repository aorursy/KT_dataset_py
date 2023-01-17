import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

df = pd.read_csv('/kaggle/input/covid19-google-community-mobility-update-sept-2020/covid19_mobility_report_sept_2020/regional/2020_ID_Region_Mobility_Report.csv')
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
jkt = df2[df2['sub_region_1'] == 'Jakarta']
jkt = jkt.groupby(['month','sub_region_1']).mean().reset_index()
jkt
jkt = jkt.drop(columns=['sub_region_1'])

jkt = jkt.set_index('month')

jkt
# create valid markers from mpl.markers

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if

item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

 

# valid_markers = mpl.markers.MarkerStyle.filled_markers

 

markers = np.random.choice(valid_markers, jkt.shape[1], replace=False)

ax = jkt.plot(kind='bar')

for i, line in enumerate(ax.get_lines()):

    line.set_marker(markers[i])



# adding legend

ax.legend(jkt.columns, loc='best')

plt.ylabel('Percent Change from Baseline')

plt.xlabel('Month')

plt.title('Jakarta Community Mobility ')

labels = ['Feb', 'Mar', 'Apr', 'May', 'June','July','Aug','Sept']

ax.set_xticklabels(labels)

plt.tight_layout()

plt.savefig('jkt.png')

plt.show()
#Bali



bali = df2[df2['sub_region_1'] == 'Bali']

bali = bali.groupby(['month','sub_region_1']).mean().reset_index()

bali = bali.drop(columns=['sub_region_1'])

bali = bali.set_index('month')



# create valid markers from mpl.markers

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if

item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

 

# valid_markers = mpl.markers.MarkerStyle.filled_markers

 

markers = np.random.choice(valid_markers, bali.shape[1], replace=False)



ax = bali.plot(kind='bar')

for i, line in enumerate(ax.get_lines()):

    line.set_marker(markers[i])



# adding legend

ax.legend(bali.columns, loc='best')

plt.ylabel('Percent Change from Baseline')

plt.xlabel('Month')

plt.title('Bali Community Mobility ')

labels = ['Feb', 'Mar', 'Apr', 'May', 'June','July','Aug','Sept']

ax.set_xticklabels(labels)

plt.tight_layout()

plt.savefig('bali.png')

plt.show()
#jatim



jatim = df2[df2['sub_region_1'] == 'East Java']

jatim = jatim.groupby(['month','sub_region_1']).mean().reset_index()

jatim = jatim.drop(columns=['sub_region_1'])

jatim = jatim.set_index('month')



# create valid markers from mpl.markers

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if

item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

 

# valid_markers = mpl.markers.MarkerStyle.filled_markers

 

markers = np.random.choice(valid_markers, jatim.shape[1], replace=False)



ax = jatim.plot(kind='bar')

for i, line in enumerate(ax.get_lines()):

    line.set_marker(markers[i])



# adding legend

ax.legend(jatim.columns, loc='best')

plt.ylabel('Percent Change from Baseline')

plt.xlabel('Month')

plt.title('Jawa Timur Community Mobility ')

labels = ['Feb', 'Mar', 'Apr', 'May', 'June','July','Aug','Sept']

ax.set_xticklabels(labels)

plt.tight_layout()

plt.savefig('jatim.png')

plt.show()