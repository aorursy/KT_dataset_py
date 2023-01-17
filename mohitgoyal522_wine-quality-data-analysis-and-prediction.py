#importing libraries that are used in this notebook

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix as CM
#importing dataset

dataset = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

dataset.head()
dataset.shape
dataset.describe()
dataset.drop(labels='density', axis=1, inplace=True)
dataset.isnull().sum()
#Plotting boxplots to see if there are any outliers in our data (considering data betwen 25th and 75th percentile as non outlier)

fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(15, 5))

ax = ax.flatten()

index = 0

for i in dataset.columns:

  if i != 'quality':

    sns.boxplot(y=i, data=dataset, ax=ax[index])

    index +=1

plt.tight_layout(pad=0.4)

plt.show()
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(15, 5))

ax = ax.flatten()

index=0

for i in dataset.columns:

  if i != 'quality':

    sns.barplot(x='quality', y=i, data=dataset, ax=ax[index])

    index+=1

plt.tight_layout(pad=0.4)

plt.show
plt.figure(figsize=(10, 10))

sns.heatmap(dataset.corr(method='pearson'), annot=True, square=True)

plt.show()



print('Correlation of different features of our dataset with quality:')

for i in dataset.columns:

  corr, _ = pearsonr(dataset[i], dataset['quality'])

  print('%s : %.4f' %(i,corr))
#for a better view this way can be used

print('Another (more clear) view of correlations among features:\n')

dataset.corr().style.background_gradient(cmap="coolwarm")
#our dataset

dataset.head()
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

dataset['quality'] = pd.cut(dataset['quality'], bins = bins, labels = group_names)

dataset.head()
dataset['quality'] = dataset['quality'].map({'bad' : 0, 'good' : 1})

dataset.head(10)
print(dataset['quality'].value_counts())

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))

ax = ax.flatten()

print('\nVisualisation of accuracies of differnt classification models')

dataset['quality'].value_counts().plot(x=0, y=1, kind='pie', figsize=(15,5), ax=ax[0])

sns.countplot(dataset['quality'], ax=ax[1])

plt.show()
X = dataset.iloc[:, :-1]

Y = (dataset.iloc[:, 10])
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.20, random_state=0)
from sklearn.preprocessing import StandardScaler as ss

SS = ss()

X_train = SS.fit_transform(X_train)

X_test = SS.transform(X_test)
logisticRegression = LogisticRegression(solver='lbfgs', random_state=0)

logisticRegression.fit(X_train, Y_train)

Y_pred_logisticRegression = logisticRegression.predict(X_test)

Y_compare_logisticRegression = pd.DataFrame({'Actual' : Y_test, 'Predicted' : Y_pred_logisticRegression})

print(Y_compare_logisticRegression.head())

print('\nConfussion matrix:')

print(CM(Y_test, Y_pred_logisticRegression))
knn = KNN(n_neighbors=2, metric='minkowski', p=2,)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)

Y_compare_knn = pd.DataFrame({'Actual' : Y_test, 'Predicted' : Y_pred_knn})

print(Y_compare_knn.head())

print('\nConfussion matrix:')

print(CM(Y_test, Y_pred_knn))
svc = SVC(kernel='rbf', gamma='scale', random_state=0)

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)

Y_compare_svc = pd.DataFrame({'Actual' : Y_test, 'Predicted' : Y_pred_svc})

print(Y_compare_svc.head())

print('\nConfussion matrix:')

print(CM(Y_test, Y_pred_svc))
nb = GaussianNB()

nb.fit(X_train, Y_train)

Y_pred_nb = nb.predict(X_test)

Y_compare_nb = pd.DataFrame({'Actual' : Y_test, 'Predicted' : Y_pred_nb})

print(Y_compare_nb.head())

print('\nConfussion matrix:')

print(CM(Y_test, Y_pred_nb))
rfc = RFC(n_estimators=25, criterion='gini', random_state=0,)

rfc.fit(X_train, Y_train)

Y_pred_rfc = rfc.predict(X_test)

Y_compare_rfc = pd.DataFrame({'Actual' : Y_test, 'Predicted' : Y_pred_rfc})

print(Y_compare_rfc.head())

print('\nConfussion matrix:')

print(CM(Y_test, Y_pred_rfc))
#K-fold cross validation

modelNames = ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector', 'Naive Bayes', 'Random Forrest']

modelClassifiers = [logisticRegression, knn, svc, nb, rfc]

models = pd.DataFrame({'modelNames' : modelNames, 'modelClassifiers' : modelClassifiers})

counter=0

score=[]

for i in models['modelClassifiers']:

  accuracy = cross_val_score(i, X_train, Y_train, scoring='accuracy', cv=10)

  print('Accuracy of %s Classification model is %.2f' %(models.iloc[counter,0],accuracy.mean()))

  score.append(accuracy.mean())

  counter+=1
pd.DataFrame({'Model Name' : modelNames,'Score' : score}).sort_values(by='Score', ascending=True).plot(x=0, y=1, kind='bar', figsize=(15,5), title='Comparison of accuracies of differnt classification models')

plt.show()