# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_df = pd.read_json("../input/train.json")

test_df = pd.read_json("../input/test.json")

train_test = pd.concat([train_df, test_df], 0).set_index("listing_id")
# ARGUMENTS



category = "building_id"

n_fake_data = 20
df = train_test[[category, "created", "interest_level"]].copy()



df["low"] = (df["interest_level"] == "low").astype(int)

df["medium"] = (df["interest_level"] == "medium").astype(int)

df["high"] = (df["interest_level"] == "high").astype(int)



df["created"] = pd.to_datetime(df["created"]).dt.dayofyear

train_test["created"] = df["created"]



del df["interest_level"]



df.head()
interests = ["low", "medium", "high"]

priors = df[interests].sum() / df[interests].sum().sum()

priors
df = (df.

      sort_values("created").

      groupby([category, "created"]).

      agg(sum)[interests].

      reset_index("created"))
# sort on created in order to make sure we are always using historical data

# see that each day has its own count data for low/medium/high



temp = df.loc[["0"]].copy().sort_values("created")

temp.head()
# by using cumsum we essentually compute the best counts using historical data



temp[interests] = temp[interests].cumsum(0)

temp.head()
# we then shift in order to make sure that one any given day, 

# we do not know what is happening on that day



temp[interests] = temp[interests].shift().fillna(0)

temp.head()
# add the prior aka 'fake data' into the mix 



temp[interests] = temp[interests] + n_fake_data * priors

temp.head()
# then we compute the MLE with the fake data, (not sure if this counts as MAP)



n = temp[interests].sum(1)

temp[interests] = temp[interests].apply(lambda _: _/n)

temp.head()
nd1 = 1 + n_fake_data 

npriors = n_fake_data * priors 
idxs = set(df.index)

total = len(idxs)



for i, idx in enumerate(idxs):

    temp = df.loc[[idx]].copy()

    if len(temp) == 1:

        temp.loc[:,interests] = temp.loc[:,interests].fillna(0) + npriors

        temp.loc[:,interests] /= nd1

    else:

        temp.loc[:,interests] = temp.loc[:,interests].cumsum(0).shift().fillna(0) + npriors

        n = temp.loc[:,interests].sum(1)

        temp.loc[:,interests] = temp.loc[:,interests].apply(lambda _: _/n)

    df.loc[[idx]] = temp

    

    if i % 1000 == 0:

        print("completed {}/{}".format(i, total))

        

df.reset_index(category, inplace=1)
features = train_test[[category, "created"]].copy()

features["listing_id"] = train_test.index

features = pd.merge(df, features, left_on=[category, "created"], right_on=[category, "created"])

features = features.set_index("listing_id")[interests]

features.columns = [category + "_" + c for c in features.columns]
features.sample(10)
features.to_csv("meancoded_{}.csv".format(category))