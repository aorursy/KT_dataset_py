import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv("../input/train.tsv", sep = "\t")
train.head()
train.isnull().any()
train.isnull().sum()
f,ax = plt.subplots(1,2, figsize=(15,5))

sns.set(color_codes=True)

sns.countplot(x="item_condition_id", data=train, ax=ax[0])

# shipping

sns.countplot(x="shipping", data=train, ax=ax[1])
train["price"].describe()
f, ax = plt.subplots(1,2,figsize=(12,6))

sns.distplot(train["price"], ax=ax[0])

sns.boxplot(train["price"], orient="v", showfliers=False, ax=ax[1]) # without outliers
# using log(price)

f, ax = plt.subplots(1,2,figsize=(15,6))

# by item condition

sns.boxplot(x = train.item_condition_id, y = np.log(train.price+1), orient = "v", ax=ax[0])

# by shipping/no shipping

sns.boxplot(x = train.shipping, y = np.log(train.price+1), orient = "v", ax=ax[1])
train["no_brand_name"] = train["brand_name"].isnull() 



f, ax = plt.subplots(1,2,figsize=(15,6))

sns.countplot(train["no_brand_name"], ax=ax[0])

ax[0].set_title("Count of Listings with/without Brand")

sns.boxplot(x = train.no_brand_name, y = np.log(train.price+1), ax=ax[1])

ax[1].set_title("Price of Listings with/without Brand")
zero = train.loc[train["price"]==0,]

print("Number of $0 listings = " + str(len(zero)))

print("Number of brands with $0 listings = " + str(sum(zero["no_brand_name"])))
f,ax = plt.subplots(2,2,figsize=(15,10))

sns.countplot(x="item_condition_id", data=zero, ax=ax[0,0])

sns.countplot(x="shipping", data=zero, ax=ax[0,1])

sns.countplot(x="no_brand_name", data=zero, ax=ax[1,0])
list(pd.unique(zero["brand_name"]))
count = zero.groupby(["brand_name"], as_index=True).count().price.sort_values(ascending = False)



# plots

f,ax = plt.subplots(1,2,figsize=(15,5))

sns.distplot(count, ax=ax[0]) # count distribution

ax[0].set(xlabel="Number of Free Listings", ylabel="Freq")

sns.barplot(count[count>5], count[count>5].index, ax = ax[1]) # for brands with count > 5

ax[1].set(xlabel="Number of Free Listings")
from wordcloud import WordCloud



wordcloud = WordCloud(width = 1200, height = 1000).generate(" ".join(zero.item_description.astype(str)))

plt.figure(figsize = (15, 10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
brand_median = train.groupby(["brand_name"], as_index = True).median().price.sort_values(ascending = False)



# plotting the distribution of brand median 

sns.distplot(brand_median)
print("10 most expensive brands:")

brand_median[:10]
top_brands = ["Demdaco", "Auto Meter", "Proenza Schouler", "Oris"]



fig, ax = plt.subplots(2, 2, figsize = (15, 10))

for brand in range(len(top_brands)):

    b = top_brands[brand]

    wordcloud = WordCloud(max_words = 200

                         ).generate(" ".join(train["item_description"][train["brand_name"] == b].astype(str)))

    ax[int(brand/2)][brand%2].axis("off")

    ax[int(brand/2)][brand%2].imshow(wordcloud)

    ax[int(brand/2)][brand%2].set_title(b, fontsize = 30)

plt.show()
# extract categories

train["main_cat"] = train.category_name.str.extract("([^/]+)/[^/]+/[^/]+", expand=False)

train["subcat1"] = train.category_name.str.extract("[^/]+/([^/]+)/[^/]+", expand=False)

train["subcat2"] = train.category_name.str.extract("[^/]+/[^/]+/([^/]+)", expand=False)
# excluding NA group

print("Number of Main Categories = " + str(len(pd.unique(train["main_cat"]))-1))

print("Number of Sub Category 1 = " + str(len(pd.unique(train["subcat1"]))-1))

print("Number of Sub Category 2 = " + str(len(pd.unique(train["subcat2"]))-1))
list(pd.unique(train["main_cat"]))
list(pd.unique(train["subcat1"]))
list(pd.unique(train["subcat2"]))
plt.figure(figsize = (20, 8))

sns.boxplot(x = train.main_cat, y = np.log(train.price+1))
sub1 = train.groupby(["subcat1"], as_index = True).median().price.sort_values(ascending = False)

sub2 = train.groupby(["subcat2"], as_index = True).median().price.sort_values(ascending = False)



f, ax = plt.subplots(1,2,figsize=(15,5))

sns.distplot(sub1, ax=ax[0])

ax[0].set_title("Sub category 1")

sns.distplot(sub2, ax=ax[1])

ax[1].set_title("Sub category 2")
top_categories = list(sub2[sub2>60].index)

top_categories
df = train.loc[train["subcat2"].isin(top_categories),]



wordcloud = WordCloud(width = 1200, height = 1000).generate(" ".join(df.item_description.astype(str)))

plt.figure(figsize = (15, 10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()