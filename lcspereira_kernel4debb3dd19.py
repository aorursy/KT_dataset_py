# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
base = pd.read_csv ('/kaggle/input/used-cars-database/autos.csv', encoding='ISO-8859-1')
base.head(20)
base.info()
base.describe()
base['seller'].value_counts()
base['offerType'].value_counts()
base['abtest'].value_counts()
base['name'].value_counts()
base = base.drop(['abtest','offerType','seller','dateCrawled','monthOfRegistration','dateCreated','nrOfPictures','postalCode','lastSeen','name'], axis=1)
base.head()
base[['price','powerPS', 'yearOfRegistration']].boxplot(figsize=(30,30))

plt.title("Numeric values boxplot")
base[['price','powerPS', 'yearOfRegistration']].hist(figsize=(30,30), bins=20)
sns.heatmap(base.corr(), annot=True)

plt.title("Correlation Map")
plt.figure(figsize=(30,30))

sns.heatmap(base.isna())

plt.title("Nulls Map")
plt.figure(figsize=(12,8))

sns.countplot(base['vehicleType'])

plt.title("Car Count By Vehicle Type")
sns.countplot(base['gearbox'])

plt.title("Car Count by Gearbox")
base['model'].value_counts()
sns.countplot(base['fuelType'])

plt.title("Car Count By Fuel Type")
sns.countplot(base['notRepairedDamage'])

plt.title("Car Count By Not Repaired Damage Status")
base.loc[base['vehicleType'].isna(), 'vehicleType'] = 'andere'

base.loc[base['gearbox'].isna(), 'gearbox'] = 'manuell'

base.loc[base['model'].isna(), 'model'] = 'andere'

base.loc[base['fuelType'].isna(), 'fuelType'] = 'andere'

base.loc[base['notRepairedDamage'].isna(), 'notRepairedDamage'] = 'nein'
plt.figure(figsize=(30,30))

sns.heatmap(base.isna())

plt.title("Nulls Map")
base.info()
df = pd.get_dummies(base.drop(['price','powerPS','yearOfRegistration','kilometer'], axis=1))

df[['price','powerPS','yearOfRegistration','kilometer']] = base[['price','powerPS','yearOfRegistration','kilometer']]
X_train, X_test, y_train, y_test = train_test_split(df.drop('price', axis=1), df['price'], test_size=0.3)
tree = RandomForestRegressor(n_estimators=100, n_jobs=4, min_samples_leaf=10, max_depth=8, min_samples_split=10, verbose=2)

tree.fit(X_train, y_train)
preds = tree.predict(X_test)
res = pd.DataFrame({'preds': preds, 'truth': y_test})

res['diff'] = res['preds'] - res['truth']

res.describe()
plt.figure(figsize=(30,30))

sns.distplot(res['diff'], kde=False)

plt.title("Difference Between Predictions And The Ground Truth Distribution")
res.boxplot()