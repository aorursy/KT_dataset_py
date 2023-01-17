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
accounts = pd.read_csv("/kaggle/input/medium-daily-digests/accounts.csv", index_col=0)

accounts.head()
mails = pd.read_csv("/kaggle/input/medium-daily-digests/mails.csv")

mails.sample(5)
articles_mails = pd.read_csv("/kaggle/input/medium-daily-digests/articles_mails.csv")

articles_mails.sample(5)
mails.groupby("to")["id"].count()
best_in = "Best in "

best_in_length = len(best_in)

topics ={section: section[best_in_length:] for section in articles_mails["section_title"].unique() if section.startswith(best_in)}

print(f"Identified {len(topics)} unique topics")
all_original_topics = set(accounts.set_index("account").values.ravel())

all_promoted_topics = set(topics.values())



all_original_topics - all_promoted_topics
articles_mails["topic"] = articles_mails["section_title"].apply(lambda section: topics.get(section))

mail_topics = articles_mails[["mail_id", "topic"]]
topic_replacement = {}

topic_key = {}

for account_id, row in  accounts.iterrows():

    for topic_label, actual_topic in row.to_dict().items():

        topic_replacement[actual_topic] = topic_label

        topic_key[(account_id, topic_label)] = actual_topic
merged_inner = pd.merge(left=mail_topics, right=mails, left_on='mail_id', right_on='id')

merged_inner = merged_inner[~merged_inner["topic"].isna()]

grouped = merged_inner.groupby(["to","topic"]).count()[["mail_id"]].rename(columns={"mail_id":"count_per_topic"})

grouped_totals = merged_inner.groupby("to").count()[["topic"]].rename(columns={"topic":"total"})



data = grouped.join(grouped_totals, how="inner")

data["percentage_within_account"] = data["count_per_topic"] / data["total"]

data = data.reset_index()

data["topic_replacement"] = data["topic"].apply(lambda topic: topic_replacement.get(topic))

data = data.set_index(["to", "topic_replacement"])

data = data["percentage_within_account"].unstack().fillna(0)
next(data.iterrows())[1].index
import matplotlib.pyplot as plt

import seaborn as sns



fig = plt.figure(figsize=(20, 10))

axes = fig.subplots(5, 4)



for idx, (account_id, counts) in enumerate(data.iterrows()):

    row = idx // 4

    column = idx % 4

    ax = axes[row][column]

    sns.barplot(counts.index, counts.values, alpha=0.8, ax = ax)

    ax.set_ylim([0,0.5])

    tickslabels = ax.get_xticklabels()

    ax.set_xticklabels([topic_key[(account_id, topic_label.get_text())] for topic_label in tickslabels])

    ax.set_xlabel(None)



plt.tight_layout()