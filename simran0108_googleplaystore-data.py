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
data=pd.read_csv('/kaggle/input/googleplaystore-acsv/googleplaystore_a.csv')

data
data.info()
data.shape
data.describe()
data["Rating"].mean()
photo_graphy=data[data['Category'].str.contains("PHOTOGRAPHY")]

photo_graphy[['Rating']].mean()
data['Type'].value_counts()
top_reviews=data.nlargest(5,["Reviews"])

top_reviews
import matplotlib.pyplot as plt

import seaborn as sns
data.groupby("Category")["App"].count().plot(kind='bar')
data.groupby("Category")["Rating"].mean().plot(kind='bar')
plt.figure(figsize=(10,5))

chart4=sns.barplot(x="Category",y="Reviews", data=data,ci=None)

chart4.set_xticklabels(chart4.get_xticklabels(),rotation=90)
plt.figure(figsize=(10,5))

chart6=sns.countplot(x="Category", data=data, hue="Type")

chart6.set_xticklabels(chart6.get_xticklabels(),rotation=90)