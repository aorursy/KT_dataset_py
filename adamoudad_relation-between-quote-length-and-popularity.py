# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#pd.options.display.max_colwidth = 100
pd.set_option('display.max_colwidth', -1)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
quotes = pd.read_json('../input/quotes.json')
quotes.head()
# Append column containing length of quotes
quotes["Length"] = quotes['Quote'].apply(lambda x: len(x))
# quotes.sort_values(by=["Length"])[["Length", "Popularity"]].plot(x="Length", y="Popularity", figsize=(15,8))
# sorted_quotes = quotes.sort_values(by=["Length"])[["Length", "Popularity"]]
fig = plt.gcf()
fig.set_size_inches(40, 10.5, forward=True)
plt.bar(sorted_quotes["Length"].values,sorted_quotes["Popularity"].values, )
quotes['quantile_length'] = pd.qcut(quotes['Length'], q=100)
quotes.groupby('quantile_length')["Popularity"].mean().plot()
# .plot(x="quantile_length", y="Popularity", figsize=(15,8))
# 'life', 'happiness', 'love', 'truth', 'inspiration', 'humor', 'philosophy', 'science', '', 'soul', 'books', 'wisdom',
# 'knowledge', 'education', 'poetry', 'hope', 'friendship', 'writing', 'religion', 'death', 'romance', 'success', 'arts',
# 'relationship', 'motivation', 'faith', 'mind', 'god', 'funny', 'quotes', 'positive', 'purpose'
quotes[quotes["Category"] == "inspiration" ].sort_values(by=["Length"])[["Length", "Popularity"]].plot(x="Length", y="Popularity", figsize=(15,6))