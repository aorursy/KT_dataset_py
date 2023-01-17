# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

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
hillary_day_counts = hillary["day"].value_counts()

hillary_day_counts = hillary_day_counts[ hillary_day_counts.index > datetime.date(2015, 12, 31)]

hillary_day_counts.sort_index(inplace=True)



trump_day_counts = trump["day"].value_counts()

trump_day_counts = trump_day_counts[ trump_day_counts.index > datetime.date(2015, 12, 31)]

trump_day_counts.sort_index(inplace=True)
day_counts = pd.concat( [hillary_day_counts, trump_day_counts], axis=1)

day_counts.columns = ["Hillary", "Trump"]
day_counts.plot()
hillary["author"].value_counts().head(10)
trump["author"].value_counts().head(10)