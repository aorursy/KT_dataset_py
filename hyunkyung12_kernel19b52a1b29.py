# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
train_test = [train,test]
train.head()
train.info()
test.info()
train['interest_level'].value_counts()
facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'bathrooms', shade=True)
facet.add_legend()
plt.show()
facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'bathrooms', shade=True)
facet.set(xlim=(0,2))
facet.add_legend()
plt.show()
facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'bathrooms', shade=True)
facet.set(xlim=(2,6))
facet.add_legend()
plt.show()
for dataset in train_test:
    dataset.loc[ dataset['bathrooms'] <= 2, 'bathrooms'] = 2,
    dataset.loc[(dataset['bathrooms'] > 2) & (dataset['bathrooms'] <= 4), 'bathrooms'] = 1,
    dataset.loc[ dataset['bathrooms'] > 4, 'bathrooms'] = 0

facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'bedrooms', shade=True)
facet.add_legend()
plt.show()
for dataset in train_test:
    dataset.loc[ dataset['bedrooms'] <= 2, 'bedrooms'] = 0,
    dataset.loc[(dataset['bedrooms'] > 2) & (dataset['bedrooms'] <= 4), 'bedrooms'] = 1,
    dataset.loc[ dataset['bedrooms'] > 4, 'bedrooms'] = 2
sum(train['building_id']=='0')
train["created"] = pd.to_datetime(train["created"])
train["month_created"] = train["created"].dt.month
train["month_created"]
train['month_created'].value_counts()
def bar_chart(feature):
    low = train[train['interest_level']=='low'][feature].value_counts() # survived 라는 값에 대해 수를 세줌
    medium = train[train['interest_level']=='medium'][feature].value_counts()
    high = train[train['interest_level']=='high'][feature].value_counts()
    df = pd.DataFrame([low, medium, high])
    df.index = ['low','medium','high']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('month_created')
train["created"] = pd.to_datetime(train["created"])
train["date_created"] = train["created"].dt.date
cnt_srs = train['date_created'].value_counts()

plt.figure(figsize=(12,4))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')
plt.show()
train['day_of_week'] = train['created'].dt.weekday
test["created"] = pd.to_datetime(test["created"])
test['day_of_week'] = test['created'].dt.weekday
fig = plt.figure(figsize=(12,6))
ax = sns.countplot(x="day_of_week", hue="interest_level",
                   hue_order=['low', 'medium', 'high'], data=train,
                   order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']);
plt.xlabel('Day of Week');
plt.ylabel('Number of occurrences');

### Adding percents over bars
height = [p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):    
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 50,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center") 
train['created_day'] = train['created'].dt.day
test['created_day'] = test['created'].dt.day
### Iterest per Day of Week
fig = plt.figure(figsize=(12,6))
sns.countplot(x="created_day", hue="interest_level", hue_order=['low', 'medium', 'high'], data=train);
plt.xlabel('created_day');
plt.ylabel('Number of occurrences');
train["num_features"] = train["features"].apply(len)
test["num_features"] = test["features"].apply(len)
llimit = np.percentile(train.latitude.values, 1)
ulimit = np.percentile(train.latitude.values, 99)
train['latitude'].ix[train['latitude']<llimit] = llimit
train['latitude'].ix[train['latitude']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train.latitude.values, bins=50, kde=False)
plt.xlabel('latitude', fontsize=12)
plt.show()
llimit = np.percentile(train.longitude.values, 1)
ulimit = np.percentile(train.longitude.values, 99)
train['longitude'].ix[train['longitude']<llimit] = llimit
train['longitude'].ix[train['longitude']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train.longitude.values, bins=50, kde=False)
plt.xlabel('longitude', fontsize=12)
plt.show()
train['price']
facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'price', shade=True)
facet.add_legend()
plt.show()
facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'price', shade=True)
facet.set(xlim=(0,100000))
facet.add_legend()
plt.show()
facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'price', shade=True)
facet.set(xlim=(100000,200000))
facet.add_legend()
plt.show()
facet = sns.FacetGrid(train, hue = "interest_level", aspect=4)
facet.map(sns.kdeplot, 'price', shade=True)
facet.set(xlim=(0,10000))
facet.add_legend()
plt.show()
train.info()
features_drop = ['building_id', 'created', 'description', 'display_address', 'features', 'manager_id', 'photos', 'street_address', 'month_created', 'date_created']
train1 = train.drop(features_drop, axis=1)
features_drop = ['building_id', 'created', 'description', 'display_address', 'features', 'manager_id', 'photos', 'street_address']
test1 = test.drop(features_drop, axis=1)
X = train[['bathrooms','bedrooms','latitude','longitude','price','day_of_week','created_day','num_features']]

y = train1['interest_level']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
log_loss(y_val, y_val_pred)
X = test[['bathrooms','bedrooms','latitude','longitude','price','day_of_week','created_day','num_features']]

y = clf.predict_proba(X)
labels2idx = {label: i for i, label in enumerate(clf.classes_)}
labels2idx
sub = pd.DataFrame()
sub["listing_id"] = test["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, labels2idx[label]]
sub.to_csv("submission_rf.csv", index=False)

