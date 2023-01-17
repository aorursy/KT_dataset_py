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

import matplotlib.pyplot as plt

from pathlib import Path



data_dir = Path('/kaggle/input/edinburgh-inside-airbnb/airbnb-data')

data_dir.exists()
calendar = pd.read_csv(Path(data_dir, 'calendar.csv'))



listings = len(calendar.listing_id.unique())

days = len(calendar.date.unique())

print(f'The are {listings} unique listings over {days} days.')
print(f'The listings start on {calendar.date.min()} and end {calendar.date.max()}')
calendar.head()
calendar.available.value_counts()
calendar_new = calendar[['date', 'available']]

calendar_new['busy'] = calendar_new.available.map( lambda x: 0 if x == 't' else 1)



calendar_new.head()
calendar_new = calendar_new.groupby('date')['busy'].mean().reset_index()

calendar_new['date'] = pd.to_datetime(calendar_new['date'])



calendar_new.head()
import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.rcParams['agg.path.chunksize'] = 10000



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



plt.figure(figsize=(10, 5))

plt.plot(calendar_new['date'], calendar_new['busy'])

plt.title('Airbnb Edinburgh Calendar')

plt.ylabel('% busy')

plt.show();
calendar['date'] = pd.to_datetime(calendar['date'])
def get_cleaned_price(price: pd.core.series.Series) -> float:

    """ Returns a float price from a pandas Series including the currency """

    return price.str.replace('$', '').str.replace(',', '').astype(float)
calendar['price'] = get_cleaned_price(calendar['price'])

calendar['adjusted_price'] = get_cleaned_price(calendar['adjusted_price'])



calendar.head()
mean_per_month = calendar.groupby(calendar['date'].dt.strftime('%B'), sort=False)['price'].mean()
mean_per_month.plot(kind = 'barh' , figsize = (12,7))

plt.xlabel('average monthly price');
calendar['day_of_the_week'] = calendar.date.dt.weekday_name

calendar.head()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

price_week = calendar[['day_of_the_week', 'price', 'adjusted_price']]

price_week.head()
price_week = price_week.groupby(['day_of_the_week']).mean().reindex(days)

price_week
price_week.plot()

ticks = list(range(0, 7, 1))

labels = "Mon Tues Weds Thurs Fri Sat Sun".split()

plt.xticks(ticks, labels);
listings = pd.read_csv(Path(data_dir, 'listings.csv'))



print(f'There are {listings.id.nunique()} records in the listing data.')
listings.head()
list(listings)
neighbourhoods = listings.groupby(by='neighbourhood_cleansed').count()[['id']].sort_values(by='id', ascending=False).head(10)

neighbourhoods
neighbourhoods.sort_values(by='id').plot(kind='barh' , figsize = (12,7))

plt.ylabel('Neighbourhood')

plt.xlabel('Rental')
listings[['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee']].head()
prices = get_cleaned_price(listings['price'])



prices.describe()
listings['price'] = prices
max_price = listings['price'].max()

max_price
listings[listings['price'] == max_price]
prices = prices.loc[(prices <= 600) & (prices > 0)]

prices.describe()
bins=50



plt.figure(figsize=(10, 5))

plt.hist(prices.values, bins=50, edgecolor='k')

plt.ylabel('Count')

plt.xlabel('Listing price in $')

plt.title('Histogram of listing prices')



plt.show()
weekly_prices = listings['weekly_price'].dropna()

weekly_prices = weekly_prices.str.replace('[$,]', '', regex=True).astype(float)



weekly_prices.describe()
weekly_prices = weekly_prices.loc[(weekly_prices <= 4000) & (weekly_prices > 0)]

weekly_prices.describe()
bins=50



plt.figure(figsize=(10, 5))

plt.hist(weekly_prices.values, bins=bins, edgecolor='k')

plt.ylabel('Count')

plt.xlabel('Listing price in $')

plt.title('Histogram of listing prices')



plt.show()
monthly_prices = listings['monthly_price'].dropna()

monthly_prices = monthly_prices.str.replace('[$,]', '', regex=True).astype(float)



monthly_prices.describe()