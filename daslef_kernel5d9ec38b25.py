import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
data.head(10)
data.columns
data.describe()
data = data.drop(['Date'], axis=1)
data.head()
data.info()
data.RainToday
data['RainToday']
numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation','Sunshine','WindGustSpeed', 

        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',

       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm']
len(numerical_features)
numeric_data = data[numerical_features]

numeric_data.info()
numeric_data
numeric_data['MinTemp'] = numeric_data['MinTemp'].fillna(12.186)
sum(numeric_data['MinTemp'])/len(numeric_data['MinTemp'])
np.mean(numeric_data['MinTemp'])
import seaborn as sns

from matplotlib.pyplot import plot as plt
numeric_data['MinTemp'].plot(figsize=(20,12)).line()
numeric_data['MinTemp'].values.sort()
sns.distplot(numeric_data['MinTemp'])
for i in numerical_features:

    numeric_data[i] = numeric_data[i].fillna(np.nanmean(numeric_data[i]))
list(enumerate(numerical_features))
import matplotlib



fig, axes = matplotlib.pyplot.subplots(16, figsize=(12,64))



for ix,el in enumerate(numerical_features):

    sns.distplot(numeric_data[el], ax=axes[ix])
numeric_data['Cloud3pm'].value_counts().plot.bar()
data.head()
y = data['RainTomorrow']
y.head()
from sklearn.linear_model import LogisticRegression
logres = LogisticRegression()
logres.fit(X=numeric_data, y=y)
logres.coef_
for ix, el in enumerate(numerical_features):

    print(el, logres.coef_[0][ix])
y_pred = logres.predict(numeric_data)
y.values
y_pred
assert len(y.values) == len(y_pred)
errors = 0

for i in range(len(y_pred)):

    if y.values[i] != y_pred[i]:

        errors += 1



print((1-errors/len(y_pred))*100)
logres.score(numeric_data, y)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(numeric_data, y)
for ix, el in enumerate(numerical_features):

    print(el, tree.feature_importances_[ix])
tree.score(numeric_data, y)
tree = DecisionTreeClassifier(max_depth=8)

tree.fit(numeric_data, y)

tree.score(numeric_data, y)
tree.predict(numeric_data)
tree_proba = tree.predict_proba(numeric_data)
logres_proba = logres.predict_proba(numeric_data)
logres_proba
mixed_proba = (tree_proba + logres_proba) / 2
y_pred_mixed = []

for i in mixed_proba:

    if i[0] > i[1]:

        y_pred_mixed.append('No')

    else:

        y_pred_mixed.append('Yes')
y_pred_mixed = ['No' if i[0] > i[1] else 'Yes' for i in mixed_proba]
y_pred_mixed = np.array(y_pred_mixed)
errors = 0

for i in range(len(y_pred_mixed)):

    if y.values[i] != y_pred_mixed[i]:

        errors += 1



print((1-errors/len(y_pred_mixed))*100)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(numeric_data, y)
rf.score(numeric_data, y)
cat_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
cat_data = data[cat_features]
cat_data['Location'].value_counts()
sns.distplot(cat_data['Location'].value_counts())
locations = set(cat_data['Location'].values)
locations
cat_data['Adelaide'] = 0
cat_data['Adelaide'] = cat_data['Location'].apply(lambda x: int(x == 'Adelaide'))
cat_data[cat_data['Adelaide'] == 1]
for i in locations:

    cat_data[i] = cat_data['Location'].apply(lambda x: int(x == i))
cat_data
cat_data.drop(['Location'], axis=1, inplace=True)
WindGustDir_dummies = pd.get_dummies(cat_data['WindGustDir'], prefix='WindGustDir')
WindDir9am_dummies = pd.get_dummies(cat_data['WindDir9am'], prefix='WindDir9am')

WindDir3am_dummies = pd.get_dummies(cat_data['WindDir3pm'], prefix='WindDir3pm')
cat_data.drop(['WindGustDir','WindDir9am','WindDir3pm'], axis=1, inplace=True)

cat_data.head()
cat_data = pd.concat([cat_data,WindGustDir_dummies, WindDir9am_dummies, WindDir3am_dummies],axis=1)
cat_data.head()
X = pd.concat([numeric_data, cat_data], axis=1)

X.head()
X.info()
logres_full = LogisticRegression()

logres_full.fit(X, y)

logres_full.score(X, y)