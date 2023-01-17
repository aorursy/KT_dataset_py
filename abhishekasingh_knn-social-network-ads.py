# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
social_data =pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')
social_data.info()
social_data.head()
social_data.describe()
social_data.isna().sum()
plt.figure(figsize=(12,7))

sns.heatmap(social_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
X = social_data.iloc[:,[2,3]].values
X.shape
y = social_data.iloc[:,4].values
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)
print("Training Set of X:", len(X_train))

print("Testing Set of X:", len(X_test))

print("Training Set of y:", len(y_train))

print("Testing Set of y:", len(y_test))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(y_test,y_predict))
print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
error_rate = []

# Might take some time

for i in range(1,25):

    classifier = KNeighborsClassifier(n_neighbors=i)

    classifier.fit(X_train,y_train)

    pred_i = classifier.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12,8))

plt.plot(range(1,25),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')