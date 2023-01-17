# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
zomato = pd.read_csv('../input/zomato.csv')

zomato.head()
zomato.shape
zomato.columns
zomato.drop(['url','address','phone','reviews_list'],axis=1,inplace=True)
zomato.head()
zomato.rename(columns={'approx_cost(for two people)':'Approx_for_2','listed_in(type)':'Type','listed_in(city)':'City'},inplace=True)
zomato.head()


sns.heatmap(zomato.isnull())
zomato['rate'].fillna(0,inplace=True)
zomato.drop('dish_liked',inplace=True,axis=1)
zomato.head()
# plt.figure(figsize=(10,5))

g = sns.countplot(zomato['Approx_for_2'])

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.show()
