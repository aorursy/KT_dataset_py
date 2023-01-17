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

kick.head()
kick.shape
kick.dtypes
import seaborn as sns
kick["main_category"].value_counts()
import matplotlib.pyplot as plt



plt.figure(figsize=(20,10))

sns.countplot(x="main_category", data=kick)
for col in kick.columns:

    print(col)
kick["currency"].value_counts().values[1:].sum()
kick["country"].value_counts()
kick = kick.loc[(kick["currency"] == "USD") & (kick["country"] == "US")]

kick.head()
kick.shape
kick["state"].value_counts()
kick = kick.loc[(kick["state"] == "failed") | (kick["state"] == "successful")]
kick["state"].shape
kick["category"].value_counts().index
kick["main_category"].value_counts().index
kick = kick.drop(["currency", "country", "category"], axis = 1)

kick.head()
kick = kick.drop(["usd_pledged_real", "usd_goal_real"], axis=1)

kick.head()
kick = kick.drop(["usd pledged"], axis=1)

kick.head()
kick["deadline"] = pd.to_datetime(kick["deadline"])
kick["deadline"].head()
kick["launched"] = pd.to_datetime(kick["launched"])
kick["launched"].head()
kick["campaign_length"] = kick["deadline"] - kick["launched"]

kick["campaign_length"].head()
kick["campaign_length"] = kick["campaign_length"].dt.days
kick["campaign_length"].head()
kick = kick.rename(columns={"campaign_length": "campaign_days"})

kick.columns
days_count = kick["campaign_days"].value_counts()

sns.lineplot(x=days_count.index, y=days_count.values)
kick_days_cu = kick.loc[(kick["campaign_days"] > 25) & (kick["campaign_days"] < 35)]

days_count_cu = kick_days_cu["campaign_days"].value_counts()

sns.barplot(x=days_count_cu.index, y=days_count_cu.values)
kick["campaign_days"].value_counts().head()
len(kick[(kick["campaign_days"] == 29) & (kick["state"] == "successful")]) / len(kick[(kick["campaign_days"] == 29)])
len(kick[(kick["campaign_days"] != 29) & (kick["state"] == "successful")]) / len(kick[(kick["campaign_days"] != 29)])
len(kick[(kick["campaign_days"] < 29) & (kick["state"] == "successful")]) / len(kick[(kick["campaign_days"] < 29)])
len(kick[(kick["campaign_days"] > 29) & (kick["state"] == "successful")]) / len(kick[(kick["campaign_days"] > 29)])
kick.head()
kick["pledged_per_backer"] = kick["pledged"] / kick["backers"]

kick.head()
kick["pledged_per_backer"].isnull().sum()
kick[(kick["backers"] == 0) & (kick["pledged_per_backer"].notnull())].head()
kick[kick["pledged_per_backer"] == 0].head()
kick.loc[(kick["backers"] == 0), "pledged_per_backer"] = 0
kick["pledged_per_backer"].isnull().sum()
kick.dtypes
kick["pledged_per_backer"] = kick["pledged_per_backer"].astype(int)

kick.dtypes
kick[["state", "pledged_per_backer"]].groupby("state").mean()
kick["ratio_of_ppb"] = kick["pledged_per_backer"] / kick["goal"]

kick.head()
kick.drop("ratio_of_ppb", axis=1)

kick.head()
kick[kick["pledged_per_backer"] > 0][["pledged_per_backer", "state"]].groupby("state").mean()
kick[["state", "goal"]].groupby("state").mean()