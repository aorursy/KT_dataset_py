import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_path = "../input/"

train_df = pd.read_json(data_path + "train.json")

test_df = pd.read_json(data_path + "test.json")

print(train_df.shape)

print(test_df.shape)
train_df.head()
ax = sns.countplot(train_df["interest_level"])
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(8,4))

p1 = sns.countplot(train_df["bathrooms"], ax=ax0)

p2 = sns.countplot(test_df["bathrooms"], ax=ax1)
ax = sns.countplot(x="bedrooms", hue="interest_level", hue_order=["low", "medium", "high"], data=train_df)
ax = sns.distplot(train_df["price"])
plimit = np.percentile(train_df["price"], 99)

# replace outlier by 99th percentile

train_df.ix[train_df["price"] > plimit, "price"] = plimit

ax = sns.distplot(train_df["price"])
ax = sns.boxplot(x="interest_level", y="bedrooms", data=train_df)
ax = sns.violinplot(x="interest_level", y="price", data=train_df)
g = sns.pairplot(train_df[['price', 'bedrooms', 'bathrooms', 'interest_level']], hue="interest_level", hue_order=["low", "medium", "high"], size=4)
train_df["ppbed"] = train_df["price"] / train_df["bedrooms"]

train_df["ppbed"] = train_df["ppbed"].replace([np.inf, -np.inf], -1)

ax = sns.distplot(train_df["ppbed"])
ax = sns.violinplot(x="interest_level", y="ppbed", data=train_df)
cnt_manager_listings = train_df["manager_id"].value_counts()

cnt_manager_listings.name = "Listings per Manager"

len(cnt_manager_listings)
ax = sns.distplot(cnt_manager_listings)
cnt_manager_listings[:10]
levels = train_df['interest_level'].unique()



aggs = dict((il, np.sum) for il in levels)

aggs['listing_id'] = np.size



manager_levels = pd.get_dummies(train_df, columns=['interest_level'], prefix='', prefix_sep='').groupby('manager_id').agg(aggs)

for l in levels:

    manager_levels['manager_skill_' + l] = manager_levels[l] / manager_levels['listing_id']

    del manager_levels[l]



del manager_levels['listing_id']

min_lat = np.percentile(train_df["latitude"], 0.1)

max_lat = np.percentile(train_df["latitude"], 99.9)

min_long = np.percentile(train_df["longitude"], 0.1)

max_long = np.percentile(train_df["longitude"], 99.9)



print(min_lat, max_lat, min_long, max_long)

outliers = train_df[(train_df["latitude"] < min_lat) | (train_df["latitude"] > max_lat) | (train_df["longitude"] < min_long) | (train_df["longitude"] > max_long) ]

print(outliers.shape[0])

train_df.shape[0]

outliers.head()



train_df.ix[train_df["latitude"] < min_lat, "latitude"] = min_lat

train_df.ix[train_df["latitude"] > max_lat, "latitude"] = max_lat



train_df.ix[train_df["longitude"] < min_long, "longitude"] = min_long

train_df.ix[train_df["longitude"] > max_long, "longitude"] = max_long
sns.lmplot(x="latitude", y="longitude", hue="interest_level", data=train_df, fit_reg=False, size=8)
high_int_df = train_df[train_df["interest_level"] == "high"]

g = sns.jointplot(x="latitude", y="longitude", data=high_int_df, size=8)
train_df["created"] = pd.to_datetime(train_df["created"])

train_df["created_date"] = train_df["created"].dt.date

train_df["created_weekday"] = train_df["created"].dt.weekday_name

train_df.dtypes
fig, ax = plt.subplots(figsize=(12,4))



cnt_date = train_df['created_date'].value_counts()

ax.bar(cnt_date.index, cnt_date.values)

ax.set_title('Rental Listing Creation Dates')

ax.set_ylabel('Number of created listings')



from matplotlib.dates import DateFormatter

ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d (%a)'))

fig.autofmt_xdate()
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

ax = sns.countplot(x="created_weekday", hue="interest_level", order=weekdays, hue_order=["low", "medium", "high"], data=train_df)