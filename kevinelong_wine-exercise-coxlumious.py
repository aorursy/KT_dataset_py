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

#show shape
#show head
#show tail
#show info
# country value counts for first ten as plot bar

from collections import OrderedDict

from operator import itemgetter



dictionary = {}



for country in wine_reviews.country.unique():

    filt = wine_reviews.country == country

    point = wine_reviews[filt].points.mean()

    dictionary[country] = point



dictionary = OrderedDict(sorted(dictionary.items(), key=itemgetter(1), reverse= True))



# Plot dictionary
