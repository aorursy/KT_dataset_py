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
reviews = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')
reviews.shape
reviews.columns

reviews = reviews.drop(['Unnamed: 0'], axis=1)
reviews = reviews.rename(columns={'region_1': 'general_area', 'region_2':'specific_area'}) 

reviews.dtypes
reviews.head(5)
reviews.tail(5)
reviews.info()
reviews.describe()
reviews.hist()
reviews.sort_values('price', ascending = False).head(5)
price = reviews['price']
np.sum(price>42)
np.sum(price>100)


over100 = reviews[reviews['price']>100]
over100.hist()
np.sum(price>500)
over500 = reviews[reviews['price']>500]
over500.hist()
np.sum(price>1000)
over1000 = reviews[reviews['price']>1000]
over1000.hist()
np.sum(price>100)
#bet10_100 = (reviews[reviews['price']>=10]) & (reviews[reviews['price']<=100])
bet10_100 = reviews.query('10 < price < 100')
np.sum(price<=100) - np.sum(price<=10)
#bet10_100.mean()
bet10_100._get_numeric_data().mean()
under10 = reviews[reviews['price'].notnull()]
under10 = under10[under10['price']<10]

np.sum(price < 10)
reviews.corr()
under10.mean()
over100.mean()
over500.mean()
over1000.mean()
coNames = reviews[reviews['country'].notnull()]
uniqueCoNames = coNames.country.unique()
print(uniqueCoNames)
len(reviews['country'].unique())
coNames.country.value_counts()

coNames.country.value_counts('country')
reviews.groupby('country')['points'].mean().sort_values(ascending=False)[:10]
india = reviews[reviews.country == 'India']
india.country.count()
england = reviews[reviews.country == 'England']
england.country.count()
england.mean()
india.mean()
england.hist()

india.hist()
england.taster_name.unique()
india.taster_name.unique()
austria = reviews[reviews.country == 'Austria']
austria.taster_name.unique()
austria.country.count()
reviews.groupby('country')['points'].mean().sort_values()[:10]
uruguay = reviews[reviews.country == 'Uruguay']
uruguay.taster_name.unique()
uruguay.country.count()
uruguay.hist()
argentina = reviews[reviews.country == 'Argentina']
argentina.taster_name.unique()
argentina.country.count()
argentina.hist()