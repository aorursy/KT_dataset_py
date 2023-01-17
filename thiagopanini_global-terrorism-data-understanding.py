import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
def format_spines(ax, right_border=True):

    """

    This function sets up borders from an axis and personalize colors

    

    Input:

        Axis and a flag for deciding or not to plot the right border

    Returns:

        Plot configuration

    """    

    # Setting up colors

    ax.spines['bottom'].set_color('#CCCCCC')

    ax.spines['left'].set_color('#CCCCCC')

    ax.spines['top'].set_visible(False)

    if right_border:

        ax.spines['right'].set_color('#CCCCCC')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')

    

def count_plot(feature, df, colors='Blues_d', hue=False, ax=None, title=''):

    """

    This function plots data setting up frequency and percentage in a count plot;

    This also sets up borders and personalization.

    

    Input:

        The feature to be counted and the dataframe. Other args are optional.

    Returns:

        Count plot.

    """    

    # Preparing variables

    ncount = len(df)

    if hue != False:

        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=ax)

    else:

        ax = sns.countplot(x=feature, data=df, palette=colors, ax=ax)



    # Make twin axis

    ax2=ax.twinx()



    # Switch so count axis is on right, frequency on left

    ax2.yaxis.tick_left()

    ax.yaxis.tick_right()



    # Also switch the labels over

    ax.yaxis.set_label_position('right')

    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')



    # Setting up borders

    format_spines(ax)

    format_spines(ax2)



    # Setting percentage

    for p in ax.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

                ha='center', va='bottom') # set the alignment of the text

    

    # Final configuration

    if not hue:

        ax.set_title(df[feature].describe().name + ' Counting plot', size=13, pad=15)

    else:

        ax.set_title(df[feature].describe().name + ' Counting plot by ' + hue, size=13, pad=15)  

    if title != '':

        ax.set_title(title)       

    plt.tight_layout()
metadata = pd.read_csv('../input/global-terrorism-metadata/global_terrorism_metadata_us.txt', sep=';', index_col='attribute')

metadata.head(10)
terr = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

terr.head()
high_importance = metadata.query('importance == "high"')

high_importance
print(f'We have {len(high_importance)} attributes of "high" importance')

terr_data = terr.loc[:, high_importance.index]

terr_data.head()
terr_data.dtypes
terr_data.isnull().sum()
null_city = terr_data[terr_data.loc[:, ['city']].isnull().values]

null_city.head()
null_city['region_txt'].value_counts()
null_city['country_txt'].value_counts()
geolocator = Nominatim(user_agent="Y_BzShFZceZ_rj_t-cI13w")

location = geolocator.reverse("52.509669, 13.376294")

print(location.address)
lat_sample = null_city['latitude'].iloc[0]

long_sample = null_city['longitude'].iloc[0]

null_city.iloc[0, [5, 6, 7]]
location = geolocator.reverse(lat_sample, long_sample)

location.address
location = geolocator.reverse(str(lat_sample) + ',' + str(long_sample))

location.address
lat_10_samples = null_city['latitude'].iloc[:10].values

long_10_samples = null_city['longitude'].iloc[:10].values

coord_address = []

for lat, long in zip(lat_10_samples, long_10_samples):

    location = geolocator.reverse(str(lat) + ',' + str(long))

    coord_address.append(location.address)

coord_address[:5]
all_lat = null_city['latitude'].values

all_long = null_city['longitude'].values

coord_address = []

for lat, long in zip(all_lat, all_long):

    try:

        location = geolocator.reverse(str(lat) + ',' + str(long))

        coord_address.append(location.address)

    except:

        coord_address.append('Unknown')

        pass

coord_address[:10]
terr_data.loc[:, 'city'].fillna('Unknown', inplace=True)
terr_data.loc[:, 'natlty1_txt'].fillna('Unknown', inplace=True)

terr_data.isnull().sum()[np.r_[7, 18]]
null_analysis = metadata.loc[terr_data.columns[terr_data.isnull().any().values], :]

attribs_null = ['summary', 'corp1', 'motive', 'nperps', 'nkillter', 'nwoundte']

null_analysis = null_analysis.loc[attribs_null, 'null_percent']

labels = ['Not Null Entries', 'Null Entries']

colors = ['skyblue', 'crimson']
circles = []

for i in range(6):

    circles.append(plt.Circle((0,0), 0.75, color='white'))

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

i = 0

j = 0

k = 0

for title, size in zip(null_analysis.index, null_analysis.values):

    axs[i, j].pie((1-size, size), labels=labels, colors=colors, autopct='%1.1f%%')

    axs[i, j].set_title(title, size=18)

    p = plt.gcf()

    axs[i, j].add_artist(circles[k])

    j += 1

    k += 1

    if j == 3:

        j = 0

        i += 1

    plt.tight_layout()

plt.show()
metadata.loc[terr_data.columns].query('data_type == "qualitative"')
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 7))

count_plot('extended', terr_data, ax=axs[0,0])

count_plot('success', terr_data, ax=axs[0,1])

count_plot('suicide', terr_data, ax=axs[0,2])

count_plot('specificity', terr_data, ax=axs[1,0])

count_plot('ishostkid', terr_data, ax=axs[1,1])

count_plot('region_txt', terr_data, ax=axs[1,2])

axs[1,2].set_xticklabels(axs[1,2].get_xticklabels(), rotation=90)

plt.show()
terr_data['eventid'] = terr_data['eventid'].astype(str)

terr_data['event_date'] = terr_data['eventid'].apply(lambda x: x[:4] + '/' + x[4:6] + '/' + x[6:8])

try: 

    terr_data['event_date'] = pd.to_datetime(terr_data['event_date'])

except ValueError as error:

    print(f'ValueError: {error}')
terr_data['event_date'][:5]
terr_data.iloc[:5, [0, 1, 2, 3, -1]]
# Days higher than 31 are not correct too

len(terr_data.query('iday > 31'))
# months higher than 12?

len(terr_data.query('imonth > 12'))
# Applying transformations

terr_data['iday'] = terr_data['iday'].apply(lambda day: day + 1 if day == 0 else day)

terr_data['imonth'] = terr_data['imonth'].apply(lambda month: month + 1 if month == 0 else month)

print((terr_data['iday'] == 0).any())

print((terr_data['imonth'] == 0).any())
year = terr_data['iyear'].astype(str)

month = terr_data['imonth'].astype(str)

day = terr_data['iday'].astype(str)

terr_data['event_date'] = year + "/" + month + "/" + day

terr_data['event_date'] = pd.to_datetime(terr_data['event_date'])

terr_data.iloc[:5, np.r_[:3, -1]]
terr_data['event_date'].dtype
terr_data['day_of_week'] = terr_data['event_date'].apply(lambda x: x.dayofweek)

terr_data['day_of_week_name'] = terr_data['event_date'].dt.day_name()

terr_data.iloc[:10, np.r_[:3, -3, -2, -1]]