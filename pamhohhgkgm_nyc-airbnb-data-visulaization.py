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
import pandas as pd

airbnb=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb
airbnb.head
airbnb.dtypes
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns
airbnb.describe()
airbnb.isnull().any()
airbnb.isnull().sum()
corr = airbnb.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
airbnb.head()
airbnb.fillna({'reviews_per_month':0}, inplace=True)
airbnb.reviews_per_month.isnull().any()

airbnb.reviews_per_month.isnull().sum()
top_host=airbnb.host_id.value_counts().head(40)

top_host
top_host_check=airbnb.calculated_host_listings_count.max()

top_host_check
sns.set(rc={'figure.figsize':(10,8)})
viz_1=top_host.plot(kind='bar')

viz_1.set_title('Hosts with the most listings in NYC')

viz_1.set_ylabel('Count of listings')

viz_1.set_xlabel('Host IDs')

viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)
sub_1=airbnb.loc[airbnb['neighbourhood_group']=='Brooklyn']

price_sub1=sub_1[['price']]



sub_2=airbnb.loc[airbnb['neighbourhood_group']=='Manhattan']

price_sub2=sub_2[['price']]



sub_3=airbnb.loc[airbnb['neighbourhood_group']=='Queens']

price_sub3=sub_3[['price']]



sub_4=airbnb.loc[airbnb['neighbourhood_group']=='Staten Island']

price_sub4=sub_4[['price']]



sub_5=airbnb.loc[airbnb['neighbourhood_group']=='Bronx']

price_sub5=sub_5[['price']]



price_list_by_n=[price_sub1, price_sub2, price_sub3, price_sub4, price_sub5]



sub_6=airbnb[airbnb.price<500]

viz_2=sns.boxplot(data=sub_6, x='neighbourhood_group', y='price')

viz_2.set_title('Density distribution of Price as per neighbourhood')
airbnb.neighbourhood.value_counts().head(10)
sub_7=airbnb.loc[airbnb['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',

                 'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]
viz_3=sns.catplot(x='neighbourhood',hue='neighbourhood_group',col='room_type',data=sub_7,kind='count')

viz_3.set_xticklabels(rotation=90)
viz_4=sub_6.plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10))

viz_4.legend()
most_reviewed=airbnb.nlargest(20,'number_of_reviews')

most_reviewed