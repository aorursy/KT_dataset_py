# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
news  = pd.read_csv('../input/abcnews-date-text.csv')

print(news.headline_text.unique().size)

print(news.publish_date.unique().size)

print(news.size)

news.tail()
news.dtypes
news['publish_date'] = news['publish_date'].astype(str)

news[['2013' in x for x in news['publish_date']]].head()
import re  # for the re.IGNORECASE flag

news[news['publish_date'].str.contains('^(2012)', re.IGNORECASE, regex=True)].size
news['year'] = news['publish_date'].apply(lambda x: x[0:4])



%matplotlib inline

news.groupby(['year']).size().plot(kind='bar', figsize = (10,5))

news.groupby(['year']).size()