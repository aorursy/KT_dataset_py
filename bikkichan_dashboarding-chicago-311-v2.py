import numpy as np 
import pandas as pd 
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
# for x in os.listdir('../input/'):
#     print(x)
for x in os.listdir('../input/'):
    if 'abandoned-vehicles' in x:
        print(x)
df = pd.read_csv('../input/311-service-requests-abandoned-vehicles.csv')
df.columns = [col.replace(' ','_').lower() for col in df.columns] # remove spaces from column names and lower case
df = df.rename(columns={'how_many_days_has_the_vehicle_been_reported_as_parked?':'days'}) # rename long column name
df['days'] = df['days'].fillna(0).astype('int') # set as int
df['creation_date'] = pd.to_datetime(df['creation_date'].map(lambda x : x.split('T')[0])) # set column as datetime
# new columns
df['duplicate'] = df['status'].map(lambda x : 1 if 'Dup' in x else 0) # set duplicate column as binary
df.loc['open'] = df['status'].map(lambda x : 1 if "open" in x.lower() else 0)
# list columns 
df.columns
# find earliest similar date for all districts
# group dataset by police district - find most recent first date
first_dates = df.groupby('police_district')['creation_date'].describe()[['first']].sort_values(by='first',ascending=False)
first_dates.head()
# earliest similar date for all districts
_year, _month, _day = (first_dates.iloc[0].values[0].year,
                       first_dates.iloc[0].values[0].month,
                       first_dates.iloc[0].values[0].day)
# filter df by max min date by police district 
df_filter = df[df['creation_date']>=pd.Timestamp(year=_year,month=_month,day=_day)]
# overview of abandoned days by district
df_filter.groupby('police_district')['days'].describe()
# There are some min and max number of days that wouldnt be posible in the given time period
# total number of daysd uring period 
max_days = (datetime.datetime.now().year - _year) * 365
max_days
# remove outliers with erroneous days
df_final = df_filter[(df_filter['days']<max_days)&(df_filter['days']>0)]
df_final.head()

# overview of days and abandon vehicles by district
aggregations = {'days':{'min_days':'min','median_days':'median','max_days':'max'},
                'type_of_service_request':{'abandon_counts':'count'},
                'duplicate':{'duplicate_count':'sum'}}
df_group = df_final.groupby('police_district').agg(aggregations)
df_group.columns = df_group.columns.droplevel(level=0)
df_group.loc[:,'one_call_count'] = df_group['abandon_counts'].astype('int')-df_group['duplicate_count'].astype('int')
df_group
# count of duplicates by district
fig, ax = plt.subplots(figsize=(15,5), ncols=1, nrows=2)
sns.barplot(ax=ax[0], x=df_group.index, y='abandon_counts', data=df_group, palette='ocean')
ax[0].set_title('Abandon service order counts by district', fontsize=14)

from matplotlib.colors import ListedColormap
df_group[['one_call_count','duplicate_count']]\
    .plot(kind='bar', stacked=True, ax=ax[1], width=.75, \
          colormap=ListedColormap(sns.color_palette("ocean_r", 2)))
ax[1].legend(prop={'size': 16, 'size':10},  loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
ax[1].set_title('One call vs Duplicate call by District', fontsize=14)
sns.despine(top=True ,right=True, left=False, bottom=False)
ax[0].set_xlabel('')
ax[0].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_xticklabels(df_group.index, rotation = 0, ha="center")
plt.tight_layout()
# Distribution plot function
def plot_by_dsrt(data, ax, name):
    kn = np.arange(len(dst_list))
    for ix, dsrt in enumerate(sorted(dst_list)):
        subset = data[data['police_district'] == dsrt]
        sns.distplot(subset['days'], hist = False, kde = True,
                     kde_kws = {'linewidth': 2},
                     color = cmap(float(ix)/kn.max()),
                     label = dsrt, ax=ax)
    ax.legend(prop={'size': 16, 'size':10}, title = 'District', loc='center left', bbox_to_anchor=(1, 0.5),ncol=3)
    ax.set_title('Day abandoned by District - {}'.format(name))
    plt.xlabel('days')
    ax.set_ylabel('Density')
    sns.despine(top=True ,right=True, left=False, bottom=False);
data = df_final[df_final['days']<np.percentile(df_final['days'],90)]
cmap = plt.get_cmap("ocean")
dst_list = df_final['police_district'].unique()

# Distribution of days 
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,8), ncols=1, nrows=3, sharex=True, sharey=True)
df_list = [(data, ax1, 'All - Total'), \
           (data[data['duplicate']==0], ax2, 'One-time'), \
           (data[data['duplicate']==1], ax3, 'Duplicate')]
for tup in df_list:
    plot_by_dsrt(tup[0], tup[1], tup[2])
plt.tight_layout()
# middle location of each district to annotate charts
df_loc_group = df_final.groupby('police_district')[['x_coordinate', 'y_coordinate','status']].agg({'x_coordinate':'mean', 'y_coordinate':'mean', 'status':'count'}).reset_index()
df_loc_group.head(9)
cmap_list = ['Greys','Purples','Blues','Greens','Oranges','Reds','YlOrBr','YlOrRd','OrRd',
             'PuRd','RdPu','BuPu','GnBu','PuBu','YlGnBu','PuBuGn','BuGn','YlGn',
             'Greys','Purples','Oranges','Blues','Greens','Reds','YlOrBr','YlOrRd','terrain','RdPu','BuPu']
fig, ax = plt.subplots(figsize=(16,16))
for ix, dsrt in enumerate(sorted(dst_list)):
        subset = df_final[df_final['police_district'] == dsrt]
        plt.scatter(subset['x_coordinate'], subset['y_coordinate'],
                    c=subset['days'], marker='*',
                    cmap=cmap_list[ix], alpha=1)
for i, txt in enumerate(df_loc_group['police_district']):
    ax.annotate(txt, (df_loc_group['x_coordinate'].iloc[i], df_loc_group['y_coordinate'].iloc[i]), color="grey")
sns.despine(top=True ,right=True, left=True, bottom=True)
plt.xticks([]), plt.yticks([]);
colors = np.random.rand(df_loc_group.shape[0])
area = (0.001 * df_loc_group['status'])**2
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(df_loc_group['x_coordinate'],df_loc_group['y_coordinate'], s=area, c=colors, alpha=0.5)
for i, txt in enumerate(df_loc_group['police_district']):
    ax.annotate(txt, (df_loc_group['x_coordinate'].iloc[i], df_loc_group['y_coordinate'].iloc[i]))
plt.title('Police districts count of abandon reports')
sns.despine(top=True ,right=True, left=True, bottom=True)
plt.xticks([]), plt.yticks([]);
df_final.groupby(['x_coordinate','y_coordinate','service_request_number']).count().reset_index('service_request_number')[['status']].head()
# median days
fig, ax = plt.subplots(figsize=(6,6))
plt.scatter(df_final['x_coordinate'], df_final['y_coordinate'], \
            c=df_final['days'], cmap='BuPu', alpha=.95, \
            s=np.sqrt(np.sqrt(df_final['days'])))

for i, txt in enumerate(df_loc_group['police_district'].astype(int)):
    ax.annotate(txt, (df_loc_group['x_coordinate'].iloc[i]-1000, df_loc_group['y_coordinate'].iloc[i]-1000), color='gray')
sns.despine(top=True ,right=True, left=True, bottom=True)
plt.title('Number of abandoned days')
plt.xticks([])
plt.yticks([]);
# total count all service calls made in each coordinate
data = df_final[['x_coordinate','y_coordinate']].dropna()
fig, axes = plt.subplots(figsize=(10,5), ncols=2, nrows=1)
nbins = 50
# Hexbin
axes[0].set_title('Count of all calls')
axes[0].hexbin(x='x_coordinate', y='y_coordinate', data=data, gridsize=nbins, cmap='Purples')
# 2D Histogram
axes[1].set_title('Count of all calls')
axes[1].hist2d(x='x_coordinate', y='y_coordinate', data=data, bins=nbins, cmap='Purples')

for i, txt in enumerate(df_loc_group['police_district'].astype(int)):
    axes[0].annotate(txt, (df_loc_group['x_coordinate'].iloc[i]-1000, df_loc_group['y_coordinate'].iloc[i]-1000), color='k')
    axes[1].annotate(txt, (df_loc_group['x_coordinate'].iloc[i]-1000, df_loc_group['y_coordinate'].iloc[i]-1000), color='k')
sns.despine(top=True ,right=True, left=True, bottom=True)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].set_xticks([])
axes[1].set_yticks([]);
# total count duplicate service calls made in each coordinate
data = df_final[df_final['duplicate']==1][['x_coordinate','y_coordinate']].dropna()
fig, axes = plt.subplots(figsize=(10,5), ncols=2, nrows=1)
nbins = 50
# Hexbin
axes[0].set_title('Count of duplicate calls')
axes[0].hexbin(x='x_coordinate', y='y_coordinate', data=data, gridsize=nbins, cmap='Blues')
# 2D Histogram
axes[1].set_title('Count of duplicate calls')
axes[1].hist2d(x='x_coordinate', y='y_coordinate', data=data, bins=nbins, cmap='Blues')

for i, txt in enumerate(df_loc_group['police_district'].astype(int)):
    axes[0].annotate(txt, (df_loc_group['x_coordinate'].iloc[i]-1000, df_loc_group['y_coordinate'].iloc[i]-1000), color='k')
    axes[1].annotate(txt, (df_loc_group['x_coordinate'].iloc[i]-1000, df_loc_group['y_coordinate'].iloc[i]-1000), color='k')
sns.despine(top=True ,right=True, left=True, bottom=True)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].set_xticks([])
axes[1].set_yticks([]);
