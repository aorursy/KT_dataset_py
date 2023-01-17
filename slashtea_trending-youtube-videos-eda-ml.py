# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
fr_videos = pd.read_csv("../input/FRvideos.csv")

fr_videos.head()
fr_categories = pd.read_json("../input/FR_category_id.json")

fr_categories["items"][0]
fr_videos.shape
fr_videos.info()
fr_videos.describe()
fr_videos.isnull().any()
fr_videos.isnull().sum()
fr_videos[fr_videos["description"].isnull() == True]
fr_videos["channel_title"].value_counts().head(20)
len(fr_videos["channel_title"].unique())
thirty_most_viewed = fr_videos.groupby("channel_title")['views'].sum().sort_values(ascending=False).head(20)

thirty_most_viewed
millions_viewed_list = []



for value in thirty_most_viewed.values:

    millions_viewed_list.append(int(str(value)[:3]))
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams





plt.figure(figsize=(30,15))

sns.set()

sns.barplot(y=thirty_most_viewed.index, x=millions_viewed_list)

plt.xlabel("Channel name")

plt.ylabel("Total views in millions")

plt.title("Total views in millions per channel")
thirty_most_liked = fr_videos.groupby("channel_title")['likes'].sum().sort_values(ascending=False).head(20)

thirty_most_liked
millions_liked_list = []



for value in thirty_most_liked.values:

    millions_liked_list.append(int(str(value)[:3]))
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams





plt.figure(figsize=(30,15))

sns.set()

sns.barplot(y=thirty_most_liked.index, x=thirty_most_liked)

plt.xlabel("Channel name")

plt.ylabel("Total liked in millions")

plt.title("Total liked in millions per channel")
thirty_most_disliked = fr_videos.groupby("channel_title")['dislikes'].sum().sort_values(ascending=False).head(20)

thirty_most_disliked
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams





plt.figure(figsize=(30,15))

sns.set()

sns.barplot(y=thirty_most_disliked.index, x=thirty_most_disliked.values)

plt.xlabel("Channel name")

plt.ylabel("Total disliked in millions")

plt.title("Total disliked in millions per channel")
thirty_most_commented = fr_videos.groupby("channel_title")['comment_count'].sum().sort_values(ascending=False).head(20)

thirty_most_commented
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams





plt.figure(figsize=(30,15))

sns.set()

sns.barplot(y=thirty_most_commented.index, x=thirty_most_commented.values)

plt.xlabel("Channel name")

plt.ylabel("Total commented in millions")

plt.title("Total commented in millions per channel")
disabled_comments = fr_videos.groupby('channel_title')['comments_disabled'].count().sort_values(ascending=False).head(20)

len(disabled_comments.index)
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams





plt.figure(figsize=(30,15))

sns.set()

sns.barplot(y=disabled_comments.index, x=disabled_comments.values)

plt.xlabel("Channel name")

plt.ylabel("Total Videos with comments off")

plt.title("Total commented in millions per channel")