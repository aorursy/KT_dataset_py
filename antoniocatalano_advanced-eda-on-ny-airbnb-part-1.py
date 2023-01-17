# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np, pandas as pd, matplotlib.pyplot as plt

plt.style.use('bmh')

import warnings

warnings.filterwarnings('ignore')



pd.options.display.float_format = '{:.6f}'.format

import seaborn as sns

import matplotlib

matplotlib.rcParams.update({'font.size': 12, 'font.family': 'Verdana'})
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head(10)
print('Number of records -->', df.shape[0])

print('Number of features -->', df.shape[1])
print('Features type:')

df.dtypes
print('Number of null values for each feature:')

df.isnull().sum()
df.info()
df1 = df[['id', 'host_id', 'neighbourhood_group', 'neighbourhood', \

          'latitude', 'longitude', 'room_type','price','minimum_nights','number_of_reviews']]
df1.head(10)
df1.shape
print('Neighbourhood group:', pd.unique(df1.neighbourhood_group), '\n', 'Room type:',pd.unique(df1.room_type))
fig, ax = plt.subplots(figsize = (10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', ax = ax, s = 20, alpha = 0.2, data=df1);

plt.title('Scatterplot evidencing Airbnb listing density in New York');
groupedbyZone = df1.groupby('neighbourhood_group')

fig, ax = plt.subplots(1,2, figsize = (14,6))

sns.countplot(df1['neighbourhood_group'], ax = ax[0], linewidth=1, edgecolor='w')

sns.countplot(df1['neighbourhood_group'], hue = df1['room_type'], ax = ax[1], linewidth=1, edgecolor='w')

ax[0].set_xlabel('Borough', labelpad = 10);

ax[1].set_xlabel('Borough', labelpad = 10);

ax[0].set_ylabel('Listings number');

ax[1].set_ylabel('Listings number');

plt.tight_layout();
sns.catplot('neighbourhood_group', 'minimum_nights', hue = 'room_type', data = df1, 

            kind = 'bar', ci = None, linewidth=1, edgecolor='w', height=8.27, aspect=11.7/8.27)

plt.xlabel('Borough', fontsize = 15, labelpad = 15)

plt.xticks(fontsize = 13)

plt.ylabel('Average minimum nights per listing',fontsize = 17, labelpad = 14);
sns.catplot('neighbourhood_group', y = 'number_of_reviews', hue = 'room_type',  kind = 'bar', 

            ci = None, data = df1, linewidth=1, edgecolor='w', height=8.27, aspect=11.7/8.27)

plt.xlabel('Borough', fontsize = 15, labelpad = 15)

plt.xticks(fontsize = 13)

plt.ylabel('Average number of reviews per listing', fontsize = 17, labelpad = 14);
print(df1.price.describe(), '\n')

print('--> 98th Price percentile:',np.percentile(df1.price, 98), '$')
price98thPerc = pd.pivot_table(df1, values = ['price'], index = ['neighbourhood_group'], \

                                aggfunc = lambda x: int(np.percentile(x, 98)))

price98thPerc.rename(columns = {'price' : '98th price percentile'}, inplace = True)

#price98thPerc.iloc[:,0] = price98thPerc.iloc[:,0].map('$ {}'.format)

price98thPerc

df1_merged = pd.merge(df1, price98thPerc, left_on ='neighbourhood_group', right_on = price98thPerc.index)

df1_noPriceOutliers = df1_merged[df1_merged['price'] < df1_merged['98th price percentile']]



numberOutliers = df1.shape[0] - df1_noPriceOutliers.shape[0]

print('In all New York there are {} listing extreme prices.'. format(numberOutliers))

print('But they represents only the {} % of total listing prices.'.format(round(numberOutliers / df1.shape[0], 3)))
plt.figure(figsize=(14,10))

ax = plt.gca()

df1_noPriceOutliers.plot(kind='scatter', x='longitude', y='latitude', c='price', ax=ax, cmap=plt.get_cmap('RdBu'), colorbar=True, alpha=0.7);
print('There are {} neighbourhoods present in this dataset.'.format(len(df1.neighbourhood.unique())))
intop10  = df1[df1.neighbourhood.isin(list(df1.neighbourhood.value_counts(ascending = False).head(10).index))]

topten = intop10.neighbourhood.value_counts(ascending = False).to_frame()

topten.rename(columns = {'neighbourhood': 'number of listings'}, inplace = True)

topten
print('Fraction and Cumulative fraction of top 10 neighbourhood over total listings:')

neighweight = pd.DataFrame([intop10.neighbourhood.value_counts()*100 / df.neighbourhood.value_counts().sum(), 

             np.cumsum(intop10.neighbourhood.value_counts()*100 / df.neighbourhood.value_counts().sum())],\

                index = ['% over total listings in New York','cumulative % over total listing in New York'])

neighweight = neighweight.T

#neighweight.rename(columns = {neighweight.columns[0]:'% over total listings', neighweight.columns[1]: 'cumulative %'})

neighweight.name = 'Top 10 Neighbourhood'

neighweight = neighweight.applymap('{:.1f}'.format)

neighweight
fig, ax = plt.subplots(figsize = (10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', ax = ax, s=20, alpha = 0.4, data=df1);

sns.scatterplot(x='longitude', y='latitude', ax = ax, s=20, \

                color = 'b', label = 'Top 10 neighbourhood \nfor Airbnb listings', alpha = 0.8, data = intop10);

ax.legend(loc = 'upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1);

ax.set_title('Airbnb listings density in New York\n Top 10 dense neighbourhood in Blue');
fig, ax = plt.subplots(figsize = (10,10))



sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', ax = ax, s=20, alpha = 0.4, data=df1);

sns.scatterplot(x='longitude', y='latitude', ax = ax, s=20, color = 'b', \

                alpha = 0.8, data = intop10[intop10.neighbourhood == 'Williamsburg'], label = 'Williamsburg');

ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1);

ax.set_title('Airbnb listings density in New York\n Williamsburg is the most dense neighbourhood');
plt.figure(figsize = (8,8))

df1.price.plot(kind = 'hist', bins = 700, linewidth=1, edgecolor='white')

plt.axis([0,600,0,7000]);

plt.xlabel('Price', labelpad = 10);

plt.ylabel('Number of listings per bin-price', labelpad = 15);

plt.title('Airbnb daily price distribution in New York without outliers', pad = 15);
g = df1_noPriceOutliers.groupby(['neighbourhood_group'])

import warnings

warnings.filterwarnings('ignore')

fig, ax = plt.subplots(figsize = (16,11))

for _ , (k, group) in enumerate(g):

    #ax[i].set_title(k)

    group.price.hist(normed = False, ax = ax, bins = 40, label = k, alpha = 0.5, linewidth=1, edgecolor='white')

    ax.legend();

ax.set_title('Price Histogram for borough', fontsize = 16, pad = 18);

ax.set_xlabel('Price', fontsize = 15, labelpad = 12)

ax.set_ylabel('Frequency in absolute value', fontsize = 15, labelpad = 12);
fig, ax = plt.subplots(figsize = (14,10))

import warnings

warnings.filterwarnings('ignore')

for _ , (k, group) in enumerate(g):

    if k in ['Bronx', 'Staten Island']:

        group.price.hist(normed = False, ax = ax, bins = 40, label = k, alpha = 0.5, linewidth=1, edgecolor='white')

        ax.legend();

ax.set_title('Price Histogram for Bronx and Staten Island', fontsize = 16, pad = 18);

ax.set_xlabel('Price', fontsize = 15, labelpad = 12)

ax.set_ylabel('Frequency in absolute value', fontsize = 15, labelpad = 12);
colors = ['red','tan','blue','green','lime']

fig, ax = plt.subplots(3, 1, figsize = (18,18))

doublegrouped = df1_noPriceOutliers.groupby(['room_type','neighbourhood_group'])

for i, (name, combo) in enumerate(doublegrouped):

    if i <= 4:

        combo.price.plot(kind = 'hist', ax = ax[0], bins = 40, 

                         label = name, alpha = 0.5, linewidth=1, edgecolor='white');

        ax[0].legend()

        ax[0].set_title('Entire home / apt')

    elif 5 <= i <= 9:

        combo.price.plot(kind = 'hist', ax = ax[1], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');

        ax[1].legend()

        ax[1].set_title('Private room')

    else:

        combo.price.plot(kind = 'hist', ax = ax[2], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');

        ax[2].legend()

        ax[2].set_title('Shared room')

for i in range(3):

    ax[i].set_ylabel('Frequency in absolute value', fontsize = 15, labelpad = 14)

plt.suptitle('Price histograms by room type and borough', fontsize = 20);
fig, ax = plt.subplots(3, 1, figsize = (16,16))

doublegrouped = df1_noPriceOutliers.groupby(['room_type','neighbourhood_group'])

for i, (name, combo) in enumerate(doublegrouped):

    if i <= 4 and name[1] in ['Bronx', 'Staten Island']:

        combo.price.plot(kind = 'hist', ax = ax[0], bins = 40, 

                         label = name, alpha = 0.5, linewidth=1, edgecolor='white');

        ax[0].legend()

        ax[0].set_title('Entire home / apt')

    elif 5 <= i <= 9 and name[1] in ['Bronx', 'Staten Island']:

        combo.price.plot(kind = 'hist', ax = ax[1], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');

        ax[1].legend()

        ax[1].set_title('Private room')

    elif i > 9 and name[1] in ['Bronx', 'Staten Island']:

        combo.price.plot(kind = 'hist', ax = ax[2], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');

        ax[2].legend()

        ax[2].set_title('Shared room')

for k in ax:

    k.set_ylabel('Listings number by bin-price', labelpad = 12)

plt.suptitle('Price histogram by room type for Bronx and Staten Island', fontsize = 15);
from scipy.stats import iqr #let's import the interquartile range function from scipy
by_room_type = pd.pivot_table(df1, values = ['price'], index = ['room_type','neighbourhood_group'], aggfunc = {"price":[np.median, np.count_nonzero, iqr]})

subtables = []

for row in by_room_type.index.levels[0]:

    subtables.append(by_room_type.loc[[row]].sort_values(by = ('price','median'), ascending = False))

by_room_type = pd.concat(t for t in subtables)



by_room_type[('price','median')] = by_room_type[('price','median')].map('$ {:.0f}'.format)

by_room_type[('price','iqr')] = by_room_type[('price','iqr')].map('$ {:.0f}'.format)

by_room_type[('price','count_nonzero')] = by_room_type[('price','count_nonzero')].map(int)



by_room_type.columns.set_levels(['number listings','IQR','median price'],level=1,inplace=True)

by_room_type.columns = by_room_type.columns.droplevel(0)

by_room_type = by_room_type[['median price', 'IQR','number listings']] # change the column order

by_room_type