import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_summary import DataFrameSummary 

from wordcloud import WordCloud

sns.set(color_codes=True)



train = pd.read_csv("../input/train.tsv", sep = "\t") 
# extract categories

train["main_cat"] = train.category_name.str.extract("([^/]+)/[^/]+/[^/]+", expand=False)

train["subcat1"] = train.category_name.str.extract("[^/]+/([^/]+)/[^/]+", expand=False)

train["subcat2"] = train.category_name.str.extract("[^/]+/[^/]+/([^/]+)", expand=False)



# check if there are missing sub-categories for listings with category names

DataFrameSummary(train.loc[pd.notnull(train["category_name"]),

                           ["category_name","main_cat","subcat1","subcat2"]]).summary()
# missing values in category_name, brand_name and item_description

train["category_name"] = train["category_name"].fillna("No Category")

train["brand_name"] = train["brand_name"].fillna("No Brand")

train["item_description"] = train["item_description"].fillna("NA") # avoid using actual words

train["main_cat"] = train["main_cat"].fillna("No Category")

train["subcat1"] = train["subcat1"].fillna("No Category")

train["subcat2"] = train["subcat2"].fillna("No Category")



# check if there are still missing values

train.isnull().sum()
print("Unique number of brands = " + str(len(pd.unique(train["brand_name"]))))
# 4809 unique brand names

brand_mean = train.groupby(["brand_name"], as_index = True).mean().price

brand_std = train.groupby(["brand_name"], as_index = True).std().price



# dataframe of mean and std 

dist = pd.concat([brand_mean, brand_std], axis=1).reset_index()

dist.columns = ["brand_name","mean","std"]



dist.isnull().sum()
# I suspect that it's because those listings only have 1 listing, and s.d. cannot be calculated

nan_sd = list(pd.unique(dist.loc[dist["std"].isnull(),"brand_name"]))

count_of_listings = train.loc[train["brand_name"].isin(nan_sd),].groupby("brand_name").count()



# visualize the number of listings for brands with NULL values for s.d.

ax = sns.countplot(x="name", data=count_of_listings)

ax.set(xlabel="Number of Listings")

plt.show()
temp = dist.loc[dist["std"].isnull(),]



# visualizing price distribution of brands with 0 s.d.

f,ax = plt.subplots(1,2,figsize=(15,6))

sns.boxplot(temp["mean"], orient="v", ax=ax[0])

sns.boxplot(temp["mean"], orient="v", showfliers=False, ax=ax[1])

ax[0].set_title("With outliers")

ax[1].set_title("Without outliers")

plt.show()
# assign std=0 for brands with missing std values

dist["std"] = dist["std"].fillna(0)



# exporting to csv

dist.to_csv("brand_priceDist.csv", index=False)
# brands with only $0 listings

brand_mean[brand_mean==0]
# remove brands with only $0 listings 

dist = dist.loc[dist["mean"] != 0,]

dist = dist.sort_values("mean", ascending=False)



top_10_brands = dist.iloc[:10,:2]

top_10_brands
# visualize price distribution 

top_brands = train.loc[train["brand_name"].isin(list(top_10_brands["brand_name"])),]



plt.figure(figsize = (20, 8))

ax = sns.boxplot(x = top_brands.brand_name, y = np.log(top_brands.price+1))

ax.set(xlabel="Brand", ylabel="log(price)")

plt.show()
bottom_10_brands = dist.iloc[-10:,:2]

bottom_10_brands
# visualize price distribution 

bottom_brands = train.loc[train["brand_name"].isin(list(bottom_10_brands["brand_name"])),]



plt.figure(figsize = (20, 8))

ax = sns.boxplot(x = bottom_brands.brand_name, y = np.log(bottom_brands.price+1))

ax.set(xlabel="Brand", ylabel="log(price)")

plt.show()
wordcloud = WordCloud(width = 1200, height = 1000).generate(" ".join(top_brands.item_description.astype(str)))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud(width=1200,height=1000).generate(" ".join(bottom_brands.item_description.astype(str)))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
fig, ax = plt.subplots(1, 2, figsize = (15, 20))

wordcloud1 = WordCloud(max_words = 100).generate(" ".join(top_brands["name"].astype(str)))

wordcloud2 = WordCloud(max_words = 100).generate(" ".join(bottom_brands["name"].astype(str)))

ax[0].axis("off")

ax[0].imshow(wordcloud1)

ax[0].set_title("10 Most Expensive Brands", fontsize = 20)

ax[1].axis("off")

ax[1].imshow(wordcloud2)

ax[1].set_title("10 Least Expensive Brands", fontsize = 20)

plt.show()
# 113 sub-categories

cat_mean = train.groupby(["subcat1"], as_index = True).mean().price

cat_std = train.groupby(["subcat1"], as_index = True).std().price



# dataframe of mean and std 

df = pd.concat([cat_mean, cat_std], axis=1).reset_index()

df.columns = ["subcat1_name","mean","std"]



df.isnull().sum()
# export 

df.to_csv("cat_priceDist.csv", index=False)
# sort by decreasing mean

df = df.sort_values("mean", ascending=False)



top_10_cats = df.iloc[:10,:2]

top_10_cats
# visualize the price distribution

top_cat = train.loc[train["subcat1"].isin(list(top_10_cats["subcat1_name"])),]



plt.figure(figsize = (20, 8))

ax = sns.boxplot(x = top_cat.subcat1, y = np.log(top_cat.price+1))

ax.set(xlabel="Brand", ylabel="log(price)")

plt.show()
bottom_10_cats = df.iloc[-10:,:2]

bottom_10_cats
# visualize the price distribution

bottom_cat = train.loc[train["subcat1"].isin(list(bottom_10_cats["subcat1_name"])),]



plt.figure(figsize = (20, 8))

ax = sns.boxplot(x = bottom_cat.subcat1, y = np.log(bottom_cat.price+1))

ax.set(xlabel="Brand", ylabel="log(price)")

plt.show()
wordcloud = WordCloud(width=1200,height=1000).generate(" ".join(top_cat.item_description.astype(str)))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud(width=1200,height=1000).generate(" ".join(bottom_cat.item_description.astype(str)))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
fig, ax = plt.subplots(1, 2, figsize = (15, 20))

wordcloud1 = WordCloud(max_words = 100).generate(" ".join(top_cat["name"].astype(str)))

wordcloud2 = WordCloud(max_words = 100).generate(" ".join(bottom_cat["name"].astype(str)))

ax[0].axis("off")

ax[0].imshow(wordcloud1)

ax[0].set_title("10 Most Expensive Categories", fontsize = 20)

ax[1].axis("off")

ax[1].imshow(wordcloud2)

ax[1].set_title("10 Least Expensive Categories", fontsize = 20)

plt.show()
f, ax = plt.subplots(7,2,figsize=(15, 60))



top_brands = ["Demdaco", "Proenza Schouler", "MCM Worldwide", "Vitamix", "Blendtec", 

             "David Yurman", "Celine"]



for brand in range(len(top_brands)):

    b = top_brands[brand]

    df = train.loc[train["brand_name"]==b,]

    sns.boxplot(x = df.item_condition_id, y = np.log(df.price+1), orient = "v", ax=ax[brand, 0])

    sns.boxplot(x = df.shipping, y = np.log(df.price+1), orient = "v", ax=ax[brand, 1])

    ax[brand, 0].set_title(b + " Item Condition", fontsize = 20)

    ax[brand, 1].set_title(b + " Shipping", fontsize = 20)



plt.show()                              
f, ax = plt.subplots(1,2,figsize=(15, 6))



df = train.loc[train["brand_name"]=="No Brand",]

sns.boxplot(x = df.item_condition_id, y = np.log(df.price+1), orient = "v", ax=ax[0])

sns.boxplot(x = df.shipping, y = np.log(df.price+1), orient = "v", ax=ax[1])

ax[0].set_title("Item Condition", fontsize = 20)

ax[1].set_title("Shipping", fontsize = 20)

plt.show()
f, ax = plt.subplots(10,2,figsize=(15, 80))



top_cats = ["Computers & Tablets", "Cameras & Photography", "Strollers", "Bags and Purses", 

            "Women's Handbags", "Musical instruments", "TV, Audio & Surveillance", "Footwear", "Golf", "Shoes"]



for cat in range(len(top_cats)):

    c = top_cats[cat]

    df = train.loc[train["subcat1"]==c,]

    sns.boxplot(x = df.item_condition_id, y = np.log(df.price+1), orient = "v", ax=ax[cat, 0])

    sns.boxplot(x = df.shipping, y = np.log(df.price+1), orient = "v", ax=ax[cat, 1])

    ax[cat, 0].set_title(c + " Item Condition", fontsize = 20)

    ax[cat, 1].set_title(c + " Shipping", fontsize = 20)



plt.show() 
f, ax = plt.subplots(10,2,figsize=(15, 80))



cats = ["Trading Cards", "Art", "Media", "Books and Zines", "Artwork", "Children", "Quilts",

        "Magazines", "Geekery", "Paper Goods"]

    

for cat in range(len(cats)):

    c = cats[cat]

    df = train.loc[train["subcat1"]==c,]

    sns.boxplot(x = df.item_condition_id, y = np.log(df.price+1), orient = "v", ax=ax[cat, 0])

    sns.boxplot(x = df.shipping, y = np.log(df.price+1), orient = "v", ax=ax[cat, 1])

    ax[cat, 0].set_title(c + " Item Condition", fontsize = 20)

    ax[cat, 1].set_title(c + " Shipping", fontsize = 20)



plt.show() 