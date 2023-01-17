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
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
file_path = '/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

data = pd.read_csv(file_path)
data.head()
print('Size of the dataset: ', data.shape)

print('Number of feature is not very large, as compared to number of samples')
# check for values for each feature

data.isnull().sum()
# unique values count for each coloumn

data.nunique()
sns.set(rc={'figure.figsize':(20,30)})
## Data distribution for neighbourhood_group and room_type.

sns.catplot(x="neighbourhood_group", col='room_type', kind="count", data=data);
## Price compared to neighbourhood_group and room_type.

sns.catplot(x="neighbourhood_group", y="price", col="room_type", col_wrap=3,            data=data);
## How review per month affect the Price.

sns.relplot(x="reviews_per_month", y="price", data=data);
## How number of review affect the price for neighbourhood_group and room_type.

sns.relplot(x="reviews_per_month", y="price", hue='neighbourhood_group', col='room_type', 

            data=data);
## How availability and minimum nights affects the price.

sns.relplot(x="minimum_nights", y="price", hue = 'availability_365', data=data);
### Linear Relation of number of reviews vs review per month.

sns.relplot(x="reviews_per_month", y="number_of_reviews", data=data);
## How calculated_host_listings_count affecting number of reviews and price.

sns.relplot(x="calculated_host_listings_count", y="price", data=data);

sns.relplot(x="number_of_reviews", y="price", data=data);
sns.regplot(x="calculated_host_listings_count", y="price", data=data);
## Analyse data when availability is 0 or 365 , to undertand this feature in more detail.

sns.catplot(x="neighbourhood_group", y="price", col="availability_365", col_wrap=2,

            data=data.query("availability_365 == 0 or availability_365 == 365 "));
## Can evalute the listing duration based on total number of reviews and review per month.

## And compare the price with the age of the listing.

data['number_of_months'] = data['number_of_reviews']/ data['reviews_per_month']

sns.regplot(x="number_of_months", y="price", data=data);
## To analyse minimum_nights requirement for each neighbourhood_group.

sns.catplot(x="neighbourhood_group", y='minimum_nights', kind="bar", data=data);

sns.catplot(x="neighbourhood_group", y='minimum_nights', kind="box", data=data);
## Room type compared to number of review , to analyse which room type get most reviews.

sns.catplot(x="room_type", y='number_of_reviews', data=data);