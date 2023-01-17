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
kick = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")
from sklearn.model_selection import train_test_split



train, test = train_test_split(kick, test_size=0.4, random_state=42)
train.head()
train.shape[0]
train = train[(train["state"] == "successful") | (train["state"] == "failed")]

train.shape[0]
train = train[(train["goal"] <= 50000) & (train["currency"] == "USD") & (train["country"] == "US")]

train.shape[0]
train[train["state"] == "successful"].shape[0] / 146925
num_of_cats = train[["category","ID"]].groupby(["category"]).count()

num_of_cats = num_of_cats.rename(columns={"ID":"total"})

num_of_cats
suc_of_cats = train[train["state"] == "successful"][["category","ID"]].groupby(["category"]).count()

suc_of_cats = suc_of_cats.rename(columns={"ID":"successful"})

suc_of_cats
cats = pd.merge(suc_of_cats, num_of_cats, on="category")

cats
cats["success_rate"] = cats["successful"] / cats["total"]
cats.sort_values("total", ascending=False).head(10)
cats.sort_values("total", ascending=False)["success_rate"].head(10).plot.bar()
cats[cats["total"] > 1000].shape[0]
cats.sort_values("success_rate", ascending=False)
cats[cats["total"] >= 30].shape[0]
plot = cats[["success_rate", "total"]].sort_values("total")["success_rate"].plot()

plot.axes.get_xaxis().set_visible(False)
train["campaign_days"] = pd.to_datetime(train["deadline"]) - pd.to_datetime(train["launched"])

train["campaign_days"] = train["campaign_days"].dt.days + 1

train.head()
total_cdays = train[["campaign_days", "ID"]].groupby("campaign_days").count().rename(columns={"ID": "total"})

total_cdays
suc_cdays = train[train["state"]=="successful"][["campaign_days", "ID"]].groupby("campaign_days").count().rename(columns={"ID": "successful"})

suc_cdays
cdays = pd.merge(suc_cdays,total_cdays,on="campaign_days")

cdays
cdays["success_rate"] = cdays["successful"] / cdays["total"]
cdays.head()
cdays["success_rate"].plot()
cdays.sort_values("success_rate").head(10)
cdays.sort_values("success_rate", ascending=False).head(10)
cdays_data = []

cdays_index = []

for i in range(4):

    curr_range = [i*23, (i+1)*23]

    cdays_index.append(str(curr_range))

    curr_slice = cdays[curr_range[0]:curr_range[1]]

    cdays_data.append(curr_slice["successful"].sum() / curr_slice["total"].sum())

    

cdays_bin = pd.DataFrame(cdays_data, index=cdays_index, columns = ["success_rate"])

cdays_bin
cdays_bin.plot.bar()
cdays_data = []

cdays_index = ["[1, 23]"]



curr_slice = cdays[1:23]

cdays_data.append(curr_slice["successful"].sum() / curr_slice["total"].sum())



for i in range(1, 4):

    curr_range = [i*23, (i+1)*23]

    cdays_index.append(str(curr_range))

    curr_slice = cdays[curr_range[0]:curr_range[1]]

    cdays_data.append(curr_slice["successful"].sum() / curr_slice["total"].sum())

    

cdays_bin = pd.DataFrame(cdays_data, index=cdays_index, columns = ["success_rate"])

cdays_bin
one = train[["main_category", "ID"]].groupby("main_category").count().rename(columns={"ID":"total"})

two = train[train["state"] == "successful"][["main_category", "ID"]].groupby("main_category").count().rename(columns={"ID": "successful"})

main_cats = pd.merge(one, two, on="main_category")

main_cats["success_rate"] = main_cats["successful"] / main_cats["total"]
main_cats["success_rate"].plot.bar()