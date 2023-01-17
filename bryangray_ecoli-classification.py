# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Matplot Lib

import matplotlib.pyplot as plt

        

import seaborn as sns

# Sklearn Libraries

from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.
ecoli_df = pd.read_csv("/kaggle/input/ecoli-uci-dataset/ecoli.csv")



ecoli_df.head()
del ecoli_df['SEQUENCE_NAME']

ecoli_df.head()
ecoli_df.shape
ecoli_df.describe()
ecoli_df.dtypes
print(ecoli_df['SITE'].unique())

print(len(ecoli_df['SITE'].unique()))
print(ecoli_df.groupby(['SITE']).agg(['count']))
groups = ecoli_df.groupby(['SITE'])

# Plot

fig, ax = plt.subplots()

ax.margins(0.05)

for name, group in groups:

    ax.plot(group.MCG, group.GVH, marker='o', linestyle='', label=name)

ax.legend(numpoints=1, loc='upper left')



plt.show()
sns.scatterplot(x=ecoli_df.MCG, y=ecoli_df.GVH, hue=ecoli_df.SITE, style = ecoli_df.SITE)
sns.pairplot(ecoli_df)
X = ecoli_df.iloc[:,0:6]

print(X.head())

y = ecoli_df.iloc[:,7]

print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print("Size of X:", X.shape)

print("Size of y:", y.shape)

print("Size of X_train:", X_train.shape)

print("Size of y_train:", y_train.shape)

print("Size of X_test:", X_test.shape)

print("Size of y_test:", y_test.shape)

# Threshold for removing correlated variables

threshold = 0.9



# Absolute value correlation matrix

corr_matrix = X_train.corr().abs()

corr_matrix.head()
# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()
# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
rt_clf = RandomForestClassifier(criterion='gini', random_state=123)

rt_clf.fit(X_train, y_train)
rt_clf.feature_importances_
y_pred = rt_clf.predict(X_test)
print("Accuracy : {}%".format(accuracy_score(y_test, y_pred)*100))

print("Classification Report: \n",classification_report(y_test, y_pred))
rt_clf = RandomForestClassifier(criterion="entropy", random_state=123)

rt_clf.fit(X_train, y_train)
rt_clf.feature_importances_
y_pred = rt_clf.predict(X_test)
print("Accuracy : {}%".format(accuracy_score(y_test, y_pred)*100))

print("Classification Report: \n",classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report
df_clf = DecisionTreeClassifier(criterion='gini', random_state=123)

df_clf.fit(X_train, y_train)
df_clf.feature_importances_
y_pred = df_clf.predict(X_test)
print("Accuracy : {}%".format(accuracy_score(y_test, y_pred)*100))

print("Classification Report: \n",classification_report(y_test, y_pred))
df_clf = DecisionTreeClassifier(criterion='entropy', random_state=123)

df_clf.fit(X_train, y_train)
df_clf.feature_importances_
y_pred = df_clf.predict(X_test)
print("Accuracy : {}%".format(accuracy_score(y_test, y_pred)*100))

print("Classification Report: \n",classification_report(y_test, y_pred))