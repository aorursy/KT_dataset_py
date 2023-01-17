# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale, robust_scale



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.info()
df.describe()
df['target'].value_counts()
pd.crosstab(df['sex'], df['target'])
unique_count = pd.Series([len(df[a].unique()) for a in df.columns], df.columns)

display(unique_count)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(unique_count.index, unique_count, palette='RdBu')

plt.ylabel('Number of unique values')

plt.show()
fig, ax = plt.subplots(figsize=(14, 8))

sns.heatmap(df.corr(), annot=True, cmap='Blues')

plt.show()
X = df.drop('target', axis=1)

y = df['target']

display(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(accuracy_score(y_test, y_pred))

print(precision_score(y_test, y_pred))

print(recall_score(y_test, y_pred))

print(f1_score(y_test, y_pred))



results = {}

results['LogisticRegression'] = accuracy_score(y_test, y_pred)
pca = PCA()

pca.fit(X)

pca.explained_variance_ratio_
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.show()
X_pca = PCA(n_components=4).fit_transform(X)

X_pca = pd.DataFrame(X_pca)

X_pca.head()
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3)



model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(accuracy_score(y_test, y_pred))

print(precision_score(y_test, y_pred))

print(recall_score(y_test, y_pred))

print(f1_score(y_test, y_pred))



results['LogisticRegression with PCA'] = accuracy_score(y_test, y_pred)
X_scaled = pd.DataFrame(robust_scale(X), columns=X.columns)

X_scaled.head()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))

X.boxplot(ax=ax1)

ax1.set_title('Before Standardization', fontsize=20)

X_scaled.boxplot(ax=ax2)

ax2.set_title('After Standardization', fontsize=20)

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)



model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(accuracy_score(y_test, y_pred))

print(precision_score(y_test, y_pred))

print(recall_score(y_test, y_pred))

print(f1_score(y_test, y_pred))



results['LogisticRegression with Standardization'] = accuracy_score(y_test, y_pred)
for key, value in results.items():

    print('{}:\t{}'.format(key, value))