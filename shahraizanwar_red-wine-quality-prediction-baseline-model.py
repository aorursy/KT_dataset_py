# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_classif, chi2





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")



bins = (2, 6.5, 8)

group_names = [0, 1]

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)



df.head()
df.describe()
plt.figure(figsize=(12,10))

ax = sns.heatmap(df.corr(), annot=True)
features = df.iloc[:,:-1]

sns.pairplot(features)

plt.show()
x = df.quality.value_counts()

sns.barplot(['Bad','Good'],x.values)

plt.show()
X = df.iloc[:,:-1].values

y = df.iloc[:,-1:].values



scaler = StandardScaler()

X_scaled = scaler.fit(X).transform(X)
## Using ANOVA test for feature selection

feature_selector = SelectKBest(f_classif, k=6)

X_scaled = feature_selector.fit_transform(X_scaled, y.flatten())



best_features = feature_selector.get_support()

print("Best Features: {}".format(list(df.iloc[:,:-1].iloc[:,best_features].columns)))
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)



print("X_train Shape: {}".format(X_train.shape))

print("y_train Shape: {}".format(Y_train.shape))

print("X_test Shape: {}".format(X_test.shape))

print("y_test Shape: {}".format(Y_test.shape))
clf = KNeighborsClassifier()

clf.fit(X_train,Y_train.flatten())

y_expect = Y_test.flatten()

y_pred = clf.predict(X_test)



print("Report:")

print(metrics.classification_report(y_expect, y_pred, zero_division=1))