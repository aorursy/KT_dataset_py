# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# DataFrame, Dict, HashTable, SQL Table

wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

wine_reviews.index.name = "index"

wine_reviews.shape

wine_reviews.head(3)

wine_reviews.tail(3)
wine_reviews.info()
wine_reviews.describe()
plt.figure(figsize=(25,10))

import seaborn as sns

from collections import OrderedDict

from operator import itemgetter

dictionary = {}



for country in wine_reviews.country.unique():

    filt = wine_reviews.country == country

    

    point = wine_reviews[filt].points.mean()

    dictionary[country] = point



dictionary = OrderedDict(sorted(dictionary.items(), key=itemgetter(1), reverse= True))

                         

sns.barplot([k for k,v in dictionary.items()],[v for k, v in dictionary.items()])



plt.xticks(rotation=90)



plt.title("Counts of Evals of province")

plt.xlabel("Cities")

plt.ylabel("Points")

plt.show()

wine_reviews['country'].value_counts().head(10).plot.bar()
