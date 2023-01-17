import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stat

from pylab import rcParams

from mpl_toolkits.basemap import Basemap
!ls ../input/coronavirusdataset/
patients = pd.read_csv('../input/coronavirusdataset/patient.csv')

route = pd.read_csv('../input/coronavirusdataset/route.csv')

ts = pd.read_csv('../input/coronavirusdataset/time.csv', index_col='date', parse_dates=True)
plt.figure(figsize=(20,10))



plt.subplot(2,2,1)

plt.title('Acc Test', fontsize=15)

ts['acc_test'].plot(kind='area')



plt.subplot(2,2,2)

plt.title('Acc Confirmed', fontsize=15)

ts['acc_confirmed'].plot(kind='area')



plt.subplot(2,2,3)

plt.title('Acc Deceased', fontsize=15)

ts['acc_deceased'].plot(kind='area')



plt.subplot(2,2,4)

plt.title('Acc Released', fontsize=15)

ts['acc_released'].plot(kind='area')



plt.tight_layout()

plt.show()
plt.figure(figsize=(20,10))



plt.subplot(2,2,1)

plt.title('New Test', fontsize=15)

ts['new_test'].plot(kind='line')



plt.subplot(2,2,2)

plt.title('New Confirmed', fontsize=15)

ts['new_confirmed'].plot(kind='line')



plt.subplot(2,2,3)

plt.title('New Deceased', fontsize=15)

ts['new_deceased'].plot(kind='line')



plt.subplot(2,2,4)

plt.title('New Released', fontsize=15)

ts['new_released'].plot(kind='line')



plt.tight_layout()

plt.show()
plt.figure(figsize=(20,10))



plt.subplot(2,1,1)

plt.title('Ratio of People Confirmed to People Tested', fontsize=18)

(ts['acc_confirmed']/ts['acc_test']).plot(kind='bar', width=1, color='orange')

plt.ylim(0,0.1)

plt.gca().xaxis.set_major_locator(plt.NullLocator())



plt.subplot(2,1,2)

plt.title('Ratio of People Released to People Confirmed', fontsize=18)

a = (ts['acc_released']/ts['acc_confirmed']).plot(kind='bar', width=1, color='orange')

plt.xticks(rotation=45)





plt.tight_layout()

plt.show()
ts[['acc_released', 'acc_deceased']].plot(kind='area', stacked=False, figsize=(20,6))

plt.title("Agg Released and Deceased", fontsize=18)

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(patients.isnull().transpose())

plt.show()
patients['age']  = 2020 - patients['birth_year']
plt.figure(figsize=(15,10))

plt.title("Age Distribution by State")

sns.boxplot(x='state', y='age', data=patients)

sns.swarmplot(x='state', y='age', data=patients, edgecolor='black', linewidth=0.3, color='black')

plt.show()
plt.figure(figsize=(15,10))

plt.title("Age Distribution by State and Sex")

sns.boxplot(x='state', y='age', hue="sex", data=patients)

#sns.swarmplot(x='state', y='age', hue=""data=patients, edgecolor='black', linewidth=0.3, color='black')

sns.despine(offset=10, trim=True)

plt.show()
patients['release_days'] = (pd.to_datetime(patients['released_date']) - pd.to_datetime(patients['confirmed_date'])).dt.days
patients['decease_days'] = (pd.to_datetime(patients['deceased_date']) - pd.to_datetime(patients['confirmed_date'])).dt.days
plt.figure(figsize=(15,6))

plt.title('Distribution of days between confirmation and death or release')

sns.distplot(patients['release_days'],bins=10, label='Release')

sns.distplot(patients['decease_days'], bins=5, label='Decease')

plt.legend()

plt.xlabel('Days')

plt.show()
grid = sns.FacetGrid(row='region', col='state', data=patients, margin_titles=True)

grid.map(plt.hist, "age", color="steelblue", bins=5)

plt.tight_layout()
reason = patients['infection_reason'].value_counts().reset_index()



plt.figure(figsize=(18,6))



my_plot = sns.barplot(data=reason, x='index', y='infection_reason', linewidth=3, edgecolor='black')

for p in my_plot.patches:

    my_plot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

                        xytext = (-3, 5), textcoords = 'offset points')

plt.xticks(rotation=90)

plt.xlabel('Infection Reason')

plt.ylabel('Count of Patients')

plt.show()
region = patients['region'].value_counts().reset_index()
plt.figure(figsize=(18,6))

my_plot = sns.barplot(data=region, x='index', y='region', linewidth=3, edgecolor='black')

for p in my_plot.patches:

    my_plot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

                        xytext = (-3, 5), textcoords = 'offset points')

plt.xticks(rotation=90)

plt.xlabel('Infection Reason')

plt.ylabel('Count of Patients')

plt.show()
grid = sns.FacetGrid(row='sex', col='state', data=patients, margin_titles=True, sharey=False, aspect=1.3, height=5)

grid.map(sns.distplot, "contact_number", color='orange', bins=5, kde=True)

plt.tight_layout()

plt.show()