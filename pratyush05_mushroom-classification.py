import numpy as np

import pandas as pd
data = pd.read_csv('../input/mushrooms.csv')
data.head()
data.info()
pd.DataFrame(data=list(data.columns.map(lambda x: data[x].nunique(()))), index=data.columns, columns=['nunique'])
data.drop(['veil-type'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()



for col in data.columns:

    data[col] = encoder.fit_transform(data[col])



data.head()
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')
plt.figure(figsize=(16,15))

sns.heatmap(data.corr(), annot=True)
sns.countplot(x='class', data=data, palette='Set1')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))



plt.subplot(1, 3, 1)

sns.distplot(data['cap-shape'])



plt.subplot(1, 3, 2)

sns.countplot(x='cap-shape', data=data, palette='rainbow')



plt.subplot(1, 3, 3)

sns.countplot(x='cap-shape', data=data, hue='class', palette='rainbow')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='cap-surface', data=data, palette='rainbow')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='cap-surface', palette='rainbow')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='cap-color', data=data, palette='rainbow')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='cap-color', palette='rainbow')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='bruises', data=data, palette='coolwarm')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='bruises', palette='coolwarm')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='odor', data=data, palette='rainbow')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='odor', palette='rainbow')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='gill-attachment', data=data, palette='coolwarm')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='gill-attachment', palette='coolwarm')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='gill-spacing', data=data, palette='coolwarm')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='gill-spacing', palette='coolwarm')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='gill-size', data=data, palette='coolwarm')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='gill-size', palette='coolwarm')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='gill-color', data=data, palette='rainbow')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='gill-color', palette='rainbow')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='stalk-shape', data=data, palette='coolwarm')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='stalk-shape', palette='coolwarm')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



plt.subplot(1, 2, 1)

sns.countplot(x='stalk-root', data=data, palette='rainbow')



plt.subplot(1, 2, 2)

sns.countplot(x='class', data=data, hue='stalk-root', palette='rainbow')
X = data[data.columns[1:]]

y = data['class']
X.head()
y.head()
from sklearn.decomposition import PCA
var = []

for n in range(1, 21):

    pca = PCA(n_components=n)

    pca.fit(X)

    var.append(np.sum(pca.explained_variance_ratio_))
plt.figure(figsize=(10,6))

plt.plot(range(1,21), var, color='red', linestyle='dashed', marker='o', markerfacecolor='black', markersize=10)

plt.title('Variance vs. Components')

plt.xlabel('Components')

plt.ylabel('Variance')
pca = PCA(n_components=15)

X = pca.fit_transform(X)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
from tpot import TPOTClassifier
pipeline = TPOTClassifier(generations=5, population_size=20, cv=5, n_jobs=-1, verbosity=2)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
from sklearn.metrics import confusion_matrix, classification_report
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))