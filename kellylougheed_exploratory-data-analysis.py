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
AB_NYC_2019 = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
AB_NYC_2019.head()
AB_NYC_2019.describe()
import matplotlib.pyplot as plt
import seaborn as sns

# Single box plot without neighborhood groups:
# sns.boxplot(y="availability_365", data=AB_NYC_2019)
# plt.show()

# Multiple box plots by neighborhood groups
sns.boxplot(x="neighbourhood_group", y="availability_365", data=AB_NYC_2019)
plt.show()

# Themes: https://python-graph-gallery.com/104-seaborn-themes/
# What type of graph will this produce?
# Which variable will be the x-axis? On the y-axis?

sns.boxplot(x="neighbourhood_group", y="price", data=AB_NYC_2019)
plt.show()
# What type of graph will this produce?
# Which variable will be the x-axis? On the y-axis?
# How is this graph different from the previous one?

affordable_airbnbs = AB_NYC_2019.loc[AB_NYC_2019["price"] < 400]

sns.boxplot(x="neighbourhood_group", y="price", data=affordable_airbnbs)
plt.show()
# How many Airbnbs are in each neighborhood?

# Number of Airbnbs in each location
AB_NYC_2019["neighbourhood_group"].value_counts()
# The list of locations to go along with the counts
AB_NYC_2019["neighbourhood_group"].value_counts().keys()
# Bar plot of value_counts of different neighborhoods
# Original code from https://python-graph-gallery.com/1-basic-barplot/

import matplotlib.pyplot as plt
 
# Get data
counts = AB_NYC_2019["neighbourhood_group"].value_counts() # [21000, 20000, 5600]
categories = AB_NYC_2019["neighbourhood_group"].value_counts().keys() # ["Manhattan", "Brooklyn", "Queens"]
x_pos = np.arange(len(categories))

# Create bars
plt.bar(x_pos, counts)
 
# Create names on the x-axis
plt.xticks(x_pos, categories)
 
# Show graphic
plt.show()
airbnb_names = AB_NYC_2019["name"]
print(airbnb_names)
# Libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
# Create a list of word
listing_names = ""
for items in airbnb_names.iteritems():
    listing_names += str(items[1]) + " "
 
# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(listing_names)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
airbnb_hosts = AB_NYC_2019["host_name"]
print(airbnb_hosts)
# Libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
# Create a list of word
host_names = ""
for items in airbnb_hosts.iteritems():
    host_names += str(items[1]) + " "
 
# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(host_names)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()