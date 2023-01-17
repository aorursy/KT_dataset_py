# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_1 = pd.read_csv("../input/7210_1.csv")

data_2 = pd.read_csv("../input/Datafiniti_Womens_Shoes_Jun19.csv")

data_3 = pd.read_csv("../input/Datafiniti_Womens_Shoes.csv")
data_2.drop(['imageURLs', 'manufacturerNumber', 'prices.condition', 'prices.dateAdded', 'prices.merchant','asins','dimension','manufacturer','prices.isSale', 'prices.returnPolicy', 'prices.shipping'], axis=1,inplace = True)
#Visualization of statistical Data Set value

data_1.describe().plot(kind = "area",fontsize=27, figsize = (20,10), table = True,colormap="rainbow")

plt.xlabel('Statistics')

plt.ylabel('Value')

plt.title("General Data Set Values")



#Correlation matrix

corrmat = data_2.corr()

#f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(corrmat, square=True);
plt.figure(figsize=(25,15))

data_2.brand.value_counts()[:50].plot(kind='barh')
def prices(price):

    a = ''

    if (price <= 100):

        a = 'cheap'

    elif (price <= 300):

        a = 'mid-range'

    else:

        a = 'expensive'

    return a



data_2['prices.amountMax'] = data_2['prices.amountMax'].map(prices)

data_2['prices.amountMin'] = data_2['prices.amountMin'].map(prices)
fig, ax = plt.subplots(figsize=(25,12))

data_2.groupby(['brand','prices.amountMax']).count()['prices.availability'].unstack().plot(ax=ax)

plt.title("Brand(Alphabtically) and its product price vs availability")
plt.figure(figsize=(10, 6))

plt.title("Price of product vs availability")

sns.barplot(y=data_2['prices.availability'],x=data_2['prices.amountMax'],hue= data_2['prices.amountMin'])