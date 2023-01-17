import pandas as pd

import numpy as np

import missingno as mso

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
train = pd.read_csv("../input/train.tsv", sep="\t")

test = pd.read_csv("../input/test.tsv", sep="\t")
print("train: {:,} rows; {} columns".format(train.shape[0], train.shape[1]))

print("test: {:,} rows; {} columns".format(test.shape[0], test.shape[1]))
train.info()
test.info()
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
mso.matrix(train.drop(["train_id", "price"], axis=1))

mso.matrix(test.drop("test_id", axis=1))
train["price"].describe()
price_log = np.log10(train["price"] + 1)

sns.distplot(price_log)

plt.show()
fig = plt.figure(figsize=(10,10))

plt.scatter(train["train_id"], np.log10(train["price"] + 1), s=1, alpha=0.5)

plt.show()
train["name_length"] = train["name"].str.len()

test["name_length"] = test["name"].str.len()



plt.figure(figsize=(7,7))

sns.distplot(train["name_length"], label="train")

sns.distplot(test["name_length"], label="test")

plt.legend()

plt.show()
train[train["name"].str.len() > 40]
def has_price(name):

    if "[rm]" in name:

        return 1

    else:

        return 0

        

train["has_price"] = train["name"].apply(lambda x: has_price(x))

test["has_price"] = test["name"].apply(lambda x: has_price(x))
pd.pivot_table(

    data=train[train["name_length"] > 40],

    values="name",

    index=["name_length", "has_price"],

    aggfunc=lambda x: len(x.unique())

)
pd.pivot_table(

    data=test[test["name_length"] > 40],

    values="name",

    index=["name_length", "has_price"],

    aggfunc=lambda x: len(x.unique())

)
train[["name", "name_length"]][train["name_length"] >= 43 ]
plt.figure(figsize=(7,7))

plt.scatter(train["name_length"], np.log(train["price"] +1 ), s=1, alpha=0.5)

plt.xlabel("name length")

plt.ylabel("price")

plt.show()
sns.distplot(train["item_condition_id"], label="train")

sns.distplot(test["item_condition_id"], label="test")

plt.legend()

plt.show()
plt.scatter(train["item_condition_id"], np.log(train["price"] + 1), s=1, alpha=0.3)

plt.xlabel("item condition id")

plt.ylabel("log price")
condition_price = []

labels = []

for c in range(5):

    condition_price.append(np.log(train[train["item_condition_id"] == (c + 1)]["price"] + 1))

    labels.append(c + 1)

    

plt.boxplot(condition_price, showmeans=True)

plt.xlabel("item condition id")

plt.ylabel("log price")

plt.show()
train_categories = set(train["category_name"].unique())

test_categories = set(test["category_name"].unique())

test_categories - train_categories
def get_category(name, level):

    try:

        cat = name.split("/")[level - 1]

    except: 

        cat = "Missing"

        

    return cat



for c in range (3):

    train["category_" + str(c + 1)] = train["category_name"].apply(lambda x: get_category(x, c + 1))

    test["category_" + str(c + 1)] = test["category_name"].apply(lambda x: get_category(x, c + 1))
print("train:")

print("category 1: {} items".format(len(train["category_1"].unique())))

print("category 2: {} items".format(len(train["category_2"].unique())))

print("category 3: {} items".format(len(train["category_3"].unique())))



print("\ntest:")

print("category 1: {} items".format(len(test["category_1"].unique())))

print("category 2: {} items".format(len(test["category_2"].unique())))

print("category 3: {} items".format(len(test["category_3"].unique())))
train_category_1 = set(train["category_1"].unique())

train_category_2 = set(train["category_2"].unique())

train_category_3 = set(train["category_3"].unique())



test_category_1 = set(test["category_1"].unique())

test_category_2 = set(test["category_2"].unique())

test_category_3 = set(test["category_3"].unique())



print("category 1 differences {}".format(len(test_category_1 - train_category_1)))

print("category 2 differences {}".format(len(test_category_2 - train_category_2)))

print("category 3 differences {}".format(len(test_category_3 - train_category_3)))
train["brand_name"].value_counts()
train["has_brand"] = train["brand_name"].notnull().astype(int)

train["has_brand"].value_counts()
plt.figure(figsize=(7, 7))

sns.distplot(np.log10(train[train["has_brand"] == 0]["price"] + 1), label="no brand")

sns.distplot(np.log10(train[train["has_brand"] == 1]["price"] + 1), label="has brand")

plt.legend()
plt.bar([0,1], train["shipping"].value_counts())

plt.xticks([0,1])

plt.show()
plt.figure(figsize=(7, 7))

sns.distplot(np.log10(train[train["shipping"] == 0]["price"] + 1), label="buyer")

sns.distplot(np.log10(train[train["shipping"] == 1]["price"] + 1), label="seller")

plt.legend()

plt.show()
train["desc_length"] = train["item_description"].fillna("").str.len()

test["desc_length"] = test["item_description"].str.len()



plt.figure(figsize=(7,7))

sns.distplot(train["desc_length"], label="train")

sns.distplot(test["desc_length"], label="test")

plt.legend()

plt.show()
txt_f = ["category_name", "brand_name", "category_1", "category_2", "category_3"]



for f in txt_f:

    train.loc[:, f] = pd.factorize(train[f])[0]
corr = train.drop(["name", "item_description", "category_name"], axis=1).corr().mul(100).astype(int)



cg = sns.clustermap(data=corr, annot=True, fmt="d")

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()