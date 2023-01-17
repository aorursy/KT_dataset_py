# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



hillary = pd.read_csv("../input/Hillary.csv", encoding="iso-8859-1", 

                      parse_dates=["timestamp"], infer_datetime_format=True,

                      usecols = ["timestamp", "id", "link", "caption", "author", "network", "likes"])

trump = pd.read_csv("../input/Trump.csv", encoding="iso-8859-1", 

                      parse_dates=["timestamp"], infer_datetime_format=True,

                      usecols = ["timestamp", "id", "link", "caption", "author", "network", "likes"])



hillary = hillary.drop_duplicates(["id"])

trump = trump.drop_duplicates(["id"])



hillary['day'] = [datetime.date(t.year, t.month, t.day) for t in hillary["timestamp"]]

trump['day'] = [datetime.date(t.year, t.month, t.day) for t in trump["timestamp"]]
hillary.head()
h_network=hillary.groupby(['network']).count()['id'].plot(kind='bar', stacked=False)

plt.title("hilary memes by network")

plt.xlabel("networks")

plt.ylabel("number of meme")

plt.show()
t_network=trump.groupby(['network']).count()['id'].plot(kind='bar', stacked=False)

plt.title("trump memes by network")

plt.xlabel("networks")

plt.ylabel("number of meme")

plt.show()
trump.groupby(['likes']).count()['id']