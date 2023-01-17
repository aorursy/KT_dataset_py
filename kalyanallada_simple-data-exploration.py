# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_json("../input/train.json")

train.shape
train.apply(lambda x: sum(x.isnull()), axis = 0)
train.head().T
test = pd.read_json("../input/test.json")

test.shape
test.head().T
train.corr()
colormap = plt.cm.plasma

plt.figure(figsize=(6,5))

plt.title('Pearson Correlation of Features', y=1.02, size=12)

sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.figure(figsize=(5,4))

train["interest_level"].value_counts().plot(kind = 'bar', legend = True, label = 'Interest Level')

plt.ylabel('Counts')
plt.figure(figsize=(5,4))

train['bathrooms'].value_counts()

train['bathrooms'].value_counts().plot(kind = 'bar', legend = True,log = True)

plt.ylabel('Counts (log scale)')

plt.xlabel('bathrooms')
plt.figure(figsize=(5,4))

train['bedrooms'].value_counts().plot(kind = 'bar', legend = True)

plt.ylabel('Counts')
plt.figure(figsize=(8,5))

train['price'].hist(bins=100, log = True)

plt.xlabel('Price')

plt.ylabel("Counts (log scale)")
plt.figure(figsize=(8,5))

train['price'].hist(bins=100, range=(0,100000), log = True)

plt.xlabel('Price')

plt.ylabel("Counts (log scale)")
plt.figure(figsize=(8,4))

sns.countplot(x="bathrooms", hue="interest_level", log=True, data=train);

plt.legend(bbox_to_anchor=(0.8, 0.8))

plt.figure(figsize=(8,5))

#sns.countplot(x="bedrooms", hue="interest_level", log=True, data=train);

sns.countplot(x="bedrooms", hue="interest_level", data=train);

plt.legend(bbox_to_anchor=(0.8, 0.9))
plt.figure(figsize=(8,4))

sns.barplot(x="bedrooms", y = "price", hue="interest_level", data=train);
plt.figure(figsize=(6,4))

train.groupby('bedrooms')['price'].mean().plot()

plt.ylabel('Mean price')
plt.figure(figsize=(8,5))

y1 = train.loc[train['interest_level'] == "high"].groupby('bedrooms')['price'].mean()

y2 = train.loc[train['interest_level'] == "medium"].groupby('bedrooms')['price'].mean()

y3 = train.loc[train['interest_level'] == "low"].groupby('bedrooms')['price'].mean()

plt.plot(y1,marker = 's', color = 'red', label = 'High')

plt.plot(y2,marker = 's',color = 'green', label = 'Medium')

plt.plot(y3,marker = 's',color = 'blue', label = 'Low')

plt.legend(bbox_to_anchor=(0.4, 1.0))

plt.ylabel('Mean price')

plt.xlabel('Bedrooms')