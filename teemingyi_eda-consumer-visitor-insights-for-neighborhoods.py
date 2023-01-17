import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from sklearn.cluster import AgglomerativeClustering

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder

print(os.listdir("../input"))
dtype = {'census_block_group': 'object', 'date_range_start': 'int', 'date_range_end':'int',

       'raw_visit_count':'float', 'raw_visitor_count':'float', 'visitor_home_cbgs':'object',

       'visitor_work_cbgs':'object', 'distance_from_home':'float', 'related_same_day_brand':'object',

       'related_same_month_brand':'object', 'top_brands':'object', 'popularity_by_hour':'object',

       'popularity_by_day':'object'}

data = pd.read_csv('../input/visit-patterns-by-census-block-group/cbg_patterns.csv', dtype=dtype)

data.info()
data.head()
data = data.dropna(subset=['census_block_group'])

data.info()
data.loc[data.raw_visit_count.isna()].head()
data = data.dropna(subset=['raw_visitor_count'])

data.info()
data['duration'] = data['date_range_end'] - data['date_range_start']

data['duration_days'] = data['duration'] / 86400

data.head()
print(data.duration_days.describe())
dtype = {'census_block_group': 'object', 'amount_land': 'float', 'amount_water': 'float', 'latitude': 'float', 'longitude': 'float'}

geo = pd.read_csv('../input/census-block-group-american-community-survey-data/safegraph_open_census_data/safegraph_open_census_data/metadata/cbg_geographic_data.csv',dtype=dtype)

geo.info()
geo = geo.set_index('census_block_group')
data = data.join(geo, on='census_block_group')

data.info()
data = data.dropna(subset=['amount_land'])

data.info()
data.sort_values(by='longitude',ascending=False).head(3)
adj_val = data.loc[data.longitude==data.longitude.max()].longitude - 360
data.loc[data.longitude==data.longitude.max(),'longitude'] = adj_val
def plot_map(data, column, alpha=0.7, colormap='coolwarm'):

    plt.figure(figsize=(30,30))



    m = Basemap(projection='lcc', 

                resolution='h',

            lat_0=38,

            lon_0=-101,

                llcrnrlon=-125, llcrnrlat=20,

                urcrnrlon=-64, urcrnrlat=47)



    m.drawcoastlines(linewidth=1)

    m.drawcountries(linewidth=2, color='blue')

    m.drawstates()

    m.drawmapboundary()

    

    lons = data["longitude"].values.tolist()

    lats = data["latitude"].values.tolist()



    # Draw scatter plot with all CBGs

    x,y = m(lons, lats)

    m.scatter(x, y, c=data[column], alpha=alpha, cmap=colormap, s=4)

    m.colorbar(location="bottom", pad="4%")

    

    plt.show()
plot_map(data,'raw_visit_count')
cluster_data = data.loc[:,['latitude','longitude']]
c_sample = cluster_data.sample(frac=0.2, random_state=100)

HC = AgglomerativeClustering(n_clusters=12,linkage='ward')

c_sample_label = HC.fit_predict(c_sample[['latitude','longitude']])
c_sample['label_HC'] = c_sample_label

c_sample['label_HC'] = c_sample['label_HC'].astype(int)

c_sample.plot(x='longitude',y='latitude', c='label_HC', kind='scatter', colormap='Paired', s=3, figsize=(20,10))

plt.show()
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=100)

RF.fit(c_sample[['latitude','longitude']], c_sample['label_HC'])
cluster_data['label'] = RF.predict(cluster_data)
cluster_data.plot(x='longitude',y='latitude', c='label', kind='scatter', colormap='Paired', s=3, figsize=(20,10))

plt.show()
data = pd.concat([data, cluster_data['label']], axis=1)
plot_map(data,'label', colormap='Paired', alpha=0.9)
data['top_brands'] = data.top_brands.map(lambda x:eval(x))
brand_dict = {}

for x in data.top_brands:

    for y in x:

        if y.lower() in brand_dict.keys():

            brand_dict[y.lower()] += 1

        else:

            brand_dict[y.lower()] = 1
top_10_pop = pd.Series(brand_dict).sort_values().tail(10)

top_10_pop.plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Brands across all CBGs')

plt.show()
(top_10_pop/pd.Series(brand_dict).sum()).plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Brands across all CBGs')

plt.xlabel('Percent of total count')

plt.show()
brand_cluster = {}

for label in data.label.unique():

    brand_cluster[label] = {}

    for x in data.loc[data.label==label].top_brands:

        for y in x:

            if y.lower() in brand_cluster[label].keys():

                brand_cluster[label][y.lower()] += 1

            else:

                brand_cluster[label][y.lower()] = 1
labels = [0, 1, 2, 3, 7, 8, 11]

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,20), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
labels = [10]

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,2.5), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster[label]).sort_values().tail(5)

    

    ax.barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster[label]).sum(), color='darkblue', alpha=0.5)

    

    ax.set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    ax.set_xlabel('Percent of total occurrence')



plt.show()
labels = [4, 5, 6, 9]

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
data['related_same_day_brand'] = data.related_same_day_brand.map(lambda x:eval(x))
brand_day_dict = {}

for x in data.related_same_day_brand:

    for y in x:

        if y.lower() in brand_day_dict.keys():

            brand_day_dict[y.lower()] += 1

        else:

            brand_day_dict[y.lower()] = 1
top_10_pop_day = pd.Series(brand_day_dict).sort_values().tail(10)

top_10_pop_day.plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Related Day Brands across all CBGs')

plt.show()
(top_10_pop_day/pd.Series(brand_day_dict).sum()).plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Related Day Brands across all CBGs')

plt.xlabel('Percent of total count')

plt.show()
brand_cluster_day = {}

for label in data.label.unique():

    brand_cluster_day[label] = {}

    for x in data.loc[data.label==label].related_same_day_brand:

        for y in x:

            if y.lower() in brand_cluster_day[label].keys():

                brand_cluster_day[label][y.lower()] += 1

            else:

                brand_cluster_day[label][y.lower()] = 1
labels = [0,1,11]

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
labels = [4,8,10]

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
labels = [3,7,9]

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
labels = [2,5,6]

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
data['related_same_month_brand'] = data.related_same_month_brand.map(lambda x:eval(x))



brand_month_dict = {}

for x in data.related_same_month_brand:

    for y in x:

        if y.lower() in brand_month_dict.keys():

            brand_month_dict[y.lower()] += 1

        else:

            brand_month_dict[y.lower()] = 1
top_10_pop_month = pd.Series(brand_month_dict).sort_values().tail(10)

top_10_pop_month.plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Related Month Brands across all CBGs')

plt.show()
(top_10_pop_month/pd.Series(brand_month_dict).sum()).plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Related Month Brands across all CBGs')

plt.xlabel('Percent of total count')

plt.show()
brand_cluster_month = {}

for label in data.label.unique():

    brand_cluster_month[label] = {}

    for x in data.loc[data.label==label].related_same_month_brand:

        for y in x:

            if y.lower() in brand_cluster_month[label].keys():

                brand_cluster_month[label][y.lower()] += 1

            else:

                brand_cluster_month[label][y.lower()] = 1
labels = np.arange(12)

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,30), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster_month[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_month[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
labels=np.arange(12)

top_day = {}

for i, label in enumerate(labels):

    

    top_day[label] = [key for key, value in brand_cluster_day[label].items() if value == max(brand_cluster_day[label].values())][0]



top_month = {}

for i, label in enumerate(labels):

    

    top_month[label] = [key for key, value in brand_cluster_month[label].items() if value == max(brand_cluster_month[label].values())][0]
data['top_related_day'] = data.label.map(lambda x: top_day[x])

data['top_related_month'] = data.label.map(lambda x: top_month[x])
OE = OrdinalEncoder()

data['top_day_encoded'] = OE.fit_transform(np.array(data['top_related_day']).reshape(-1,1))

data['top_month_encoded'] = OE.fit_transform(np.array(data['top_related_month']).reshape(-1,1))
plot_map(data, 'top_day_encoded', colormap='Paired')
plot_map(data, 'top_month_encoded', colormap='Paired')
data['number_of_top_brand'] = data.top_brands.map(len)

data['number_of_related_day_brand'] = data.related_same_day_brand.map(len)

data['number_of_related_month_brand'] = data.related_same_month_brand.map(len)
data['intersection_day_top_brand'] = data.apply(lambda x: list(set(x.related_same_day_brand)&set(x.top_brands)), axis=1)
data['intersection_month_top_brand'] = data.apply(lambda x: list(set(x.related_same_month_brand)&set(x.top_brands)), axis=1)
brand_day_dict_intersect = {}

for x in data.intersection_day_top_brand:

    for y in x:

        if y.lower() in brand_day_dict_intersect.keys():

            brand_day_dict_intersect[y.lower()] += 1

        else:

            brand_day_dict_intersect[y.lower()] = 1

            

brand_month_dict_intersect = {}

for x in data.intersection_month_top_brand:

    for y in x:

        if y.lower() in brand_month_dict_intersect.keys():

            brand_month_dict_intersect[y.lower()] += 1

        else:

            brand_month_dict_intersect[y.lower()] = 1
top_10_pop_day_int = pd.Series(brand_day_dict_intersect).sort_values().tail(10)

top_10_pop_day_int.plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Popular and Related Day Brands across all CBGs')

plt.show()
(top_10_pop_day_int/pd.Series(brand_day_dict_intersect).sum()).plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Popular and Related Day Brands across all CBGs')

plt.xlabel('Percent of total count')

plt.show()
brand_cluster_day_int = {}

for label in data.label.unique():

    brand_cluster_day_int[label] = {}

    for x in data.loc[data.label==label].intersection_day_top_brand:

        for y in x:

            if y.lower() in brand_cluster_day_int[label].keys():

                brand_cluster_day_int[label][y.lower()] += 1

            else:

                brand_cluster_day_int[label][y.lower()] = 1
labels = np.arange(12)

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,30), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster_day_int[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day_int[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
top_10_pop_month_int = pd.Series(brand_month_dict_intersect).sort_values().tail(10)

top_10_pop_month_int.plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Popular and Related Month Brands across all CBGs')

plt.show()
(top_10_pop_month_int/pd.Series(brand_month_dict_intersect).sum()).plot(kind='barh', color='darkblue', alpha=0.5)

plt.title('Top 10 Popular and Related Month Brands across all CBGs')

plt.xlabel('Percent of total count')

plt.show()
brand_cluster_month_int = {}

for label in data.label.unique():

    brand_cluster_month_int[label] = {}

    for x in data.loc[data.label==label].intersection_month_top_brand:

        for y in x:

            if y.lower() in brand_cluster_month_int[label].keys():

                brand_cluster_month_int[label][y.lower()] += 1

            else:

                brand_cluster_month_int[label][y.lower()] = 1
labels = np.arange(12)

fig, ax = plt.subplots(nrows=len(labels), figsize=(5,30), sharex='all')



for i, label in enumerate(labels):

    

    top_5_pop = pd.Series(brand_cluster_month_int[label]).sort_values().tail(5)

    

    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_month_int[label]).sum(), color='darkblue', alpha=0.5)

    

    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')

    

    if i == len(labels)-1:

        ax[i].set_xlabel('Percent of total occurrence')



plt.show()
plt.figure(figsize=(10,5))

data.distance_from_home.plot(kind='hist',bins=50)

plt.show()
data['distance_log'] = np.log(data['distance_from_home'])
plt.figure(figsize=(10,10))

sns.scatterplot(x='raw_visit_count',y='distance_from_home', data=data)

plt.show()
data['popularity_by_day'] = data.popularity_by_day.map(lambda x:eval(x))
data = pd.concat([data, pd.DataFrame.from_dict(data['popularity_by_day'].to_dict(), orient='index')],axis=1)

data.head()
data[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].agg('median').plot(kind='bar')

plt.show()
data['day_sum'] = data['Monday']+data['Tuesday']+data['Wednesday']+data['Thursday']+data['Friday']+data['Saturday']+data['Sunday']
for x in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:

    data[x+'_prop'] = data[x]/data['day_sum']
plt.figure(figsize=(10,8))

sns.violinplot(x='variable', y='value', data=pd.melt(data[['census_block_group','Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']], id_vars=['census_block_group']))

plt.xticks(rotation=60)

plt.show()
data.loc[data[['Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']].agg('max', axis=1)>0.5].sort_values(by='day_sum',ascending=False).head()
data.loc[data[['Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']].agg('max', axis=1)>0.5].sort_values(by='day_sum',ascending=False).head(1)[['Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']].plot(kind='bar')

plt.show()
data['popularity_by_hour'] = data.popularity_by_hour.map(lambda x:eval(x))
data = pd.concat([data, pd.DataFrame.from_records(data['popularity_by_hour'].to_dict()).transpose()], axis=1)

data.head()
plt.figure(figsize=(10,5))

data[np.arange(24)].agg('median').plot(kind='bar', color='grey')

plt.xticks(rotation=0)

plt.show()
data['hour_sum'] = data[np.arange(24)].sum(axis=1)
for x in np.arange(24):

    data[str(x)+'_prop'] = data[x]/data['hour_sum']
plt.figure(figsize=(20,10))

sns.violinplot(x='variable', y='value', data=pd.melt(data[['census_block_group']+[str(x)+'_prop' for x in np.arange(24)]], id_vars=['census_block_group']))

plt.xticks(rotation=90)

plt.show()
data.loc[data[[str(x)+'_prop' for x in np.arange(24)]].agg('max', axis=1)>0.2].sort_values(by='hour_sum',ascending=False).head()
data.loc[data[[str(x)+'_prop' for x in np.arange(24)]].agg('max', axis=1)>0.2].sort_values(by='hour_sum',ascending=False).head(1)[[str(x)+'_prop' for x in np.arange(24)]].plot(kind='bar')

plt.legend().remove()

plt.show()
data.raw_visit_count.plot(kind='hist',bins=20)

plt.show()
data.raw_visitor_count.plot(kind='hist',bins=20)

plt.show()
data['percent_unique_visitor'] = data['raw_visitor_count']/data['raw_visit_count']

data['percent_unique_visitor'].plot(kind='hist',bins=50)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x='raw_visit_count',y='raw_visitor_count', data=data)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(y='raw_visit_count',x='percent_unique_visitor', data=data)

plt.show()
data['raw_visit_count_log'] = np.log(data['raw_visit_count'])

data['raw_visitor_count_log'] = np.log(data['raw_visitor_count'])
plt.figure(figsize=(10,10))

sns.scatterplot(x='raw_visit_count_log',y='percent_unique_visitor', data=data)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(y='distance_from_home',x='percent_unique_visitor', data=data)

plt.show()
data['visitor_home_cbgs'] = data.visitor_home_cbgs.map(lambda x:eval(x))

data['visitor_work_cbgs'] = data.visitor_work_cbgs.map(lambda x:eval(x))
data['number_of_home_cbgs'] = data.visitor_home_cbgs.map(len)

data['number_of_work_cbgs'] = data.visitor_work_cbgs.map(len)
home_cbgs = {}

for x in data.visitor_home_cbgs:

    for key, value in x.items():

        if key in home_cbgs.keys():

            home_cbgs[key] += np.array([1, value])

        else:

            home_cbgs[key] = np.array([1, value])
home_cbgs_df = pd.DataFrame.from_dict(home_cbgs, orient='index',columns=['number_of_occurrence_home','total_visitors_count_home'])
work_cbgs = {}

for x in data.visitor_work_cbgs:

    for key, value in x.items():

        if key in work_cbgs.keys():

            work_cbgs[key] += np.array([1, value])

        else:

            work_cbgs[key] = np.array([1, value])

            

work_cbgs_df = pd.DataFrame.from_dict(work_cbgs, orient='index',columns=['number_of_occurrence_work','total_visitors_count_work'])
total = pd.concat([home_cbgs_df,work_cbgs_df],axis=1, sort=True)

total.fillna(0, inplace=True)

total.head()
total['log_home'] = np.log(total.total_visitors_count_home+1)

total['log_work'] = np.log(total.total_visitors_count_work+1)
plt.figure(figsize=(10,10))

sns.scatterplot(x='total_visitors_count_home',y='total_visitors_count_work', data=total)

plt.show()
total.log_home.plot(kind='hist',bins=60)
plt.figure(figsize=(10,10))

sns.scatterplot(x='log_home',y='log_work', data=total)

plt.show()
data = data.join(total, on='census_block_group')
data.loc[:,total.columns] = data.loc[:,total.columns].fillna(0)
data.loc[(data.number_of_occurrence_home!=0)&(data.number_of_home_cbgs!=0)]
plt.figure(figsize=(10,10))

sns.regplot(x='number_of_work_cbgs',y='number_of_occurrence_work', data=data.loc[(data.number_of_occurrence_work!=0)&(data.number_of_work_cbgs!=0)])

plt.show()