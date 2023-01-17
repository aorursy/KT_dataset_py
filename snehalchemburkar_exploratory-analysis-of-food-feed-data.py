# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
fao = pd.read_csv("../input/FAO.csv", encoding='latin1')
print("Lets look at a few rows of this data:\n")

fao.head()
print("Column names:\n", fao.columns)
print("Dataset contains {} rows and {} columns".format(fao.shape[0], fao.shape[1]))
print(fao.dtypes)
fao.Element.unique()
sns.catplot('Element', data = fao, kind = 'count');
print("The feed and food data is available for {} countries listed below:\n".format(len(fao.Area.unique())))

fao.Area.unique()
# The count plot for 174 countries won't be to clear hence lets look at the values here

print("Area wise row counts:\n", fao.Area.value_counts())
# To get the top 20 food/feed production for Y2013

fao_top_20 = fao.nlargest(20, 'Y2013')

fao_top_20.head()
# Let's look at the largest producers in 2013

fao_top_20.Area.unique()
fao_top_20.Element.unique()
# Let's plot the the top 20 productions for food and feed for 2013

sns.catplot(x= 'Area', data = fao_top_20, kind ='count', height = 6, aspect = 1.5);
# Let's take a look  at the food items in the dataset

print(fao.Item.unique())

# Let's look for the top 5 producers for each item in 2013

fao_wheat = fao[fao.Item == 'Wheat and products']

print(fao_wheat.head())
fao_wheat_area = fao_wheat.groupby(['Area'])['Y2013'].sum()
fao_wheat_area.shape