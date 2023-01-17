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
# import data

data = pd.read_csv("../input/googleplaystore.csv")

#data info

data.info()
# data top

data.head(10)

#find all categories script

categories = []

data["Category"]

for category in data["Category"]:

    if category not in categories:

        categories.append(category)

print(categories)

print(len(categories))
#data's the frequency of rating

data.Rating.plot(kind="hist",bins=50,figsize=(5,5))


plt.xlim([0,5])

for cat in categories[0:5]:

    data[data["Category"] == cat].Rating.plot(kind="hist",bins=50,figsize=(10,10),label=cat,alpha=0.5,linewidth=2)

    plt.title(cat)

    plt.legend(loc="upper left")
plt.xlim([0,5])



for cat in categories[5:10]:

    data[data["Category"] == cat].Rating.plot(kind="hist",bins=50,figsize=(10,10),label=cat,alpha=0.5,linewidth=2)

    plt.title(cat)

    plt.legend(loc="upper left")
plt.xlim([0,5])



for cat in categories[10:15]:

    data[data["Category"] == cat].Rating.plot(kind="hist",bins=50,figsize=(10,10),label=cat,alpha=0.5,linewidth=2)

    plt.title(cat)

    plt.legend(loc="upper left")
plt.xlim([0,5])



for cat in categories[15:20]:

    data[data["Category"] == cat].Rating.plot(kind="hist",bins=50,figsize=(10,10),label=cat,alpha=0.5,linewidth=2)

    plt.title(cat)

    plt.legend(loc="upper left")
plt.xlim([0,5])



for cat in categories[20:25]:

    data[data["Category"] == cat].Rating.plot(kind="hist",bins=50,figsize=(10,10),label=cat,alpha=0.5,linewidth=2)

    plt.title(cat)

    plt.legend(loc="upper left")
plt.xlim([0,5])



for cat in categories[25:30]:

    data[data["Category"] == cat].Rating.plot(kind="hist",bins=50,figsize=(10,10),label=cat,alpha=0.5,linewidth=2)

    plt.title(cat)

    plt.legend(loc="upper left")
plt.xlim([0,5])



for cat in categories[30:34]:

    data[data["Category"] == cat].Rating.plot(kind="hist",bins=50,figsize=(10,10),label=cat,alpha=0.5,linewidth=2)

    plt.title(cat)

    plt.legend(loc="upper left")