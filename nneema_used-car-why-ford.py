# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cars_df = pd.read_csv("/kaggle/input/craigslist-carstrucks-data/vehicles.csv")
cars_df.head()
cars_df.shape
cars_manufacture = cars_df["manufacturer"].value_counts()

print(cars_manufacture)
import seaborn as sns
top_seller=cars_df["manufacturer"].value_counts()[0:10]

sns.barplot(y=top_seller.index, x=top_seller.values);
bottom_seller=cars_df["manufacturer"].value_counts()[-10:]

sns.barplot(y=bottom_seller.index, x=bottom_seller.values, orient="h");
car_price_mean=cars_df.groupby(["manufacturer"])["price"].mean()

y_val=[]

for b,c in bottom_seller.items():

    y_val.append(car_price_mean[b])

g=sns.barplot(x=bottom_seller.index, y=y_val)

g.set_xticklabels(g.get_xticklabels(), rotation=45)
y_val=[]

for b,c in top_seller.items():

    y_val.append(car_price_mean[b])

g=sns.barplot(x=top_seller.index, y=y_val)

g.set_xticklabels(g.get_xticklabels(), rotation=45)
bottom_seller=cars_df["manufacturer"].value_counts()[-10:]

filter_cars_df = cars_df[cars_df["manufacturer"].isin(bottom_seller.index)]
(filter_cars_df

.groupby("manufacturer")["condition"]

.value_counts(normalize=True)

.mul(100)

.rename('percent')

.reset_index()

.pipe((sns.catplot,'data'), x="percent",y='manufacturer',hue="condition",kind='bar'))
top_seller=cars_df["manufacturer"].value_counts()[0:10]

filter_cars_df = cars_df[cars_df["manufacturer"].isin(top_seller.index)]

(filter_cars_df

.groupby("manufacturer")["condition"]

.value_counts(normalize=True)

.mul(100)

.rename('percent')

.reset_index()

.pipe((sns.catplot,'data'), x="percent",y='manufacturer',hue="condition",kind='bar'))
top_types = cars_df["type"].value_counts()[0:5]
filter_cars_df=cars_df[cars_df["condition"].isin(["excellent", "good", "like new"])]

filter_cars_df=filter_cars_df[filter_cars_df["price"]<25000]

filter_cars_df = filter_cars_df[filter_cars_df["type"].isin(top_types.index)]
top_seller=cars_df["manufacturer"].value_counts()[0:7]

filter_top_df = filter_cars_df[filter_cars_df["manufacturer"].isin(top_seller.index)]
(filter_top_df

.groupby("manufacturer")["type"]

.value_counts(normalize=False)

.rename('sold units')

.reset_index()

.pipe((sns.catplot,'data'), x="sold units",y='manufacturer',hue="type",kind='bar'))
filter_bottom_df = filter_cars_df[filter_cars_df["manufacturer"].isin(bottom_seller.index)]
(filter_bottom_df

.groupby("manufacturer")["type"]

.value_counts()

.rename('sold units')

.reset_index()

.pipe((sns.catplot,'data'), x="sold units",y='manufacturer',hue="type",kind='bar'))