import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

%matplotlib inline
## Read file



data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

data.head(3)
print(data.shape)

data = data.drop_duplicates(subset=['App'], keep = 'first')

print(data.shape)
data.isnull().sum()
data.info()
dt_ctg=data.groupby('Category',as_index=False)['Rating'].mean()

dt_ctg.head()
dt_gn=data.groupby('Genres',as_index=False)['Rating'].mean()

dt_gn.head(10)
data.columns=[each.replace(" ","_") for each in data.columns]

data.columns
data["Category"]=[each.replace("_"," ") for each in data.Category]

data["Price"]=[str(each.replace("$","")) for each in data.Price]
sns.countplot(x='Type',data=data)
sns.barplot(x='Type', y='Rating', data=data)
plt.figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')



# specify hue="categorical_variable"

sns.barplot(x='Content_Rating', y='Rating', hue="Type", data=data, estimator=np.median)

plt.show()
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')



# specify hue="categorical_variable"

sns.boxplot(x='Content_Rating', y='Rating', hue="Type", data=data)

plt.show()
plt.figure(figsize=(16,8))

sns.countplot(y='Category',data=data)

plt.show()