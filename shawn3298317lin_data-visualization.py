import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns
data_df = pd.read_csv("/kaggle/input/bu-cs542-fall19/data.csv")

data_df
test_ind = pd.read_csv("/kaggle/input/bu-cs542-fall19/test.csv")

train_ind = pd.read_csv("/kaggle/input/bu-cs542-fall19/train.csv")

val_ind = pd.read_csv("/kaggle/input/bu-cs542-fall19/val.csv")

print("Train set: %i, Val set: %i, Test set: %i" % (len(train_ind), len(val_ind), len(test_ind)))
train_df = data_df.merge(train_ind, on=["id"])

val_df = data_df.merge(val_ind, on=["id"])

test_df = data_df.merge(test_ind, on=["id"])

train_df
train_df["host_response_time"].unique()
train_df.info()
data_missing = train_df.isna()

data_missing = data_missing.sum()

data_missing = data_missing / len(data_df)

print("Train Missing field rate (sorted):")

data_missing[data_missing > 0.0].sort_values(ascending=False)

train_df[["square_feet"]].plot()

train_df[["price"]].plot()

# plt.scatter(train_df["square_feet"], train_df["price"])
# Unique ID

print(len(data_df["id"].unique()) == len(data_df))

len(data_df["id"].unique())
train_df["host_response_time"].value_counts()
train_df.columns
train_df["host_since_timestamp"] = train_df["host_since"].apply(lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d")) if not pd.isnull(x) else 0.0)

train_df["first_review_timestamp"] = train_df["first_review"].apply(lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d")) if not pd.isnull(x) else 0.0)

train_df["last_review_timestamp"] = train_df["last_review"].apply(lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d")) if not pd.isnull(x) else 0.0)
plt.scatter(train_df["id"], train_df["host_since_timestamp"], s=0.5, alpha=0.4)
plt.scatter(train_df["id"], train_df["first_review_timestamp"], s=0.5, alpha=0.4)

plt.scatter(train_df["id"], train_df["last_review_timestamp"], s=0.5, alpha=0.4)


# plt.scatter(train_df["id"], train_df["last_review_timestamp"], s=0.5, alpha=0.4)



train_df["host_response_rate"] = train_df["host_response_rate"].apply(lambda x: -0.1 if pd.isnull(x) else float(x.replace("%", ""))/100)

train_df["host_response_rate"].unique()
plt.scatter(train_df["id"], train_df["host_response_rate"], s=0.8, alpha=0.8)
sns.distplot(train_df["host_response_rate"].tolist(), bins=500)

plt.title("Host response rate value-frequency")

plt.xlim(-0.25, 1.1)
def process_binary_category(x):

    if pd.isnull(x):

        return -1

    elif x == "t":

        return 1

    else:

        return 0

train_df["host_is_superhost_encode"] = train_df["host_is_superhost"].apply(lambda x: process_binary_category(x))

sns.distplot(train_df["host_is_superhost_encode"].tolist(), bins=20)
train_df["host_is_superhost"].value_counts()
train_df["host_listings_count"] = train_df["host_listings_count"].apply(lambda x: -1 if pd.isnull(x) else int(x))

sns.distplot(train_df["host_listings_count"].tolist(), bins=30)
print(max(train_df["host_listings_count"].tolist()))

print(min(train_df["host_listings_count"].tolist()))
print(train_df["host_identity_verified"].unique())

print(train_df["host_identity_verified"].value_counts())

train_df["host_identity_verified_encode"] = train_df["host_identity_verified"].apply(lambda x: process_binary_category(x))

sns.distplot(train_df["host_identity_verified_encode"].tolist(), bins=30)
lats = train_df["latitude"].tolist()

longs = train_df["longitude"].tolist() 

print(max(lats), min(lats), np.median(lats), np.median(lats)-min(lats))

print(max(longs), min(longs), np.median(longs), np.median(longs)-min(longs))

img = plt.imread("/kaggle/input/shawn-airbnb-feature-eng/austin_map5.png")

plt.figure(figsize=(60, int(60/1004*1884)))

plt.imshow(img, extent=[-98.38806,-97.09717,29.97040,30.56344], alpha=0.3) 

plt.scatter(longs, lats, c="b", s=30, alpha=0.3)

# scat_plot = sns.scatterplot(lats, longs)

# scat_plot.imshow(img, extent=[-98.38806,-97.09717,29.97040,30.56344], zorder=0, alpha=0.3)
from matplotlib.colors import LinearSegmentedColormap



ncolors = 256

color_array = plt.get_cmap('viridis')(range(ncolors))

# change alpha values

color_array[:,-1] = np.linspace(1.0,0.0,ncolors)

# create a colormap object

map_object = LinearSegmentedColormap.from_list(name='viridis_alpha',colors=color_array)

# register this new colormap with matplotlib

plt.register_cmap(cmap=map_object)
from scipy.stats.kde import gaussian_kde



img = plt.imread("/kaggle/input/shawn-airbnb-feature-eng/austin_map5.png")



k = gaussian_kde(np.vstack([longs, lats]))

xi, yi = np.mgrid[-98.38806:-97.09717:0.005,29.97040:30.56344:0.005]

zi = k(np.vstack([xi.flatten(), yi.flatten()]))



fig, axs = plt.subplots(1, 1, figsize=(60, int(60/1004*1884)))

axs.imshow(img, extent=[-98.38806,-97.09717,29.97040,30.56344], alpha=0.4) 

h = axs.pcolormesh(xi, yi, zi.reshape(xi.shape), vmin=0, alpha=0.7, cmap=map_object)

train_df["is_location_exact_encode"] = train_df["is_location_exact"].apply(lambda x: process_binary_category(x))

sns.distplot(train_df["is_location_exact_encode"].tolist(), bins=10, kde=False)
train_df["property_type"].value_counts()

len(train_df["property_type"].unique())

sns.catplot(y="property_type", kind="count", data=train_df);
train_df["room_type"].value_counts()

len(train_df["room_type"].unique())

sns.catplot(y="room_type", kind="count", data=train_df);

# len(train_df["room_type"].unique())
train_df["accommodates"].value_counts()

len(train_df["accommodates"].unique())

sns.distplot(train_df["accommodates"], bins=50, kde=False)
train_df["bathrooms"].value_counts()

len(train_df["bathrooms"].unique())

sns.distplot(train_df["bathrooms"], bins=50, kde=False)
train_df["bedrooms"].value_counts()

len(train_df["bedrooms"].unique())

sns.distplot(train_df["bedrooms"], bins=50, kde=False)
train_df["beds"].value_counts()

len(train_df["beds"].unique())

sns.distplot(train_df["beds"], bins=25, kde=False)
# The relationship btw `bedrooms` and `beds`

sns.scatterplot(train_df["bedrooms"], train_df["beds"])
train_df["bed_type"].value_counts()

len(train_df["bed_type"].unique())

sns.catplot(y="bed_type", kind="count", data=train_df);

# len(train_df["room_type"].unique())
import itertools

from collections import Counter

amenities = train_df["amenities"].tolist()#.value_counts()

l = [[tok.strip() for tok in a.replace("\"", "")[1:-1].split(",") if tok.strip() != ""] for a in amenities]

plt.figure(figsize=(6, 40))

sns.countplot(y=list(itertools.chain(*l)))

train_df["square_feet"] = train_df["square_feet"].apply(lambda x: -1 if pd.isnull(x) else x)

# train_df["square_feet"].value_counts()

print("Non-nan entry:", len(train_df[train_df["square_feet"] > -1]["square_feet"]))

sns.distplot(train_df[train_df["square_feet"] > -1]["square_feet"], bins=50, kde=False)
# train_df["number_of_reviews"].unique()

print(max(train_df["number_of_reviews"]))

sns.distplot(train_df["number_of_reviews"], bins=50, kde=False)
print(max(train_df["number_of_reviews_ltm"]))

sns.distplot(train_df["number_of_reviews_ltm"], bins=50, kde=False)
print(min(train_df["review_scores_rating"]))

train_df["review_scores_rating"].unique()

train_df["review_scores_rating"] = train_df["review_scores_rating"].apply(lambda x: -1 if pd.isnull(x) else x)

sns.distplot(train_df["review_scores_rating"], bins=50, kde=False)
scores = ["review_scores_rating", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"]

sns.distplot(train_df[scores[1:]], color=["r", "g", "b", "y", "purple"], label=scores[1:], bins=15, kde=False)

plt.legend()

train_df[scores[1:]].mean()
train_df["price"].unique()

print(max(train_df["price"]))

print(min(train_df["price"]))

print(max(val_df["price"]))

print(min(val_df["price"]))

sns.distplot(train_df["price"])
