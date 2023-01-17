# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname, filename))

df
sns.barplot(y = 'fixed acidity',x = 'quality',data = df)
sns.barplot(x = 'quality',y = 'pH',data = df)
sns.barplot(x = 'quality',y = 'alcohol',data = df)
sns.barplot(x = 'quality',y = 'free sulfur dioxide',data = df)
df['quality'].value_counts()
df['quality'] = df['quality'].apply(lambda x: "Poor" if x<6.5 else "Good")
df['quality'].value_counts()
df['PoorQuality'] = pd.get_dummies(df['quality'],drop_first=True)
df.corr()
plt.figure(figsize=(14,12))

ax = sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
normalized_df=(df.drop(['quality','PoorQuality'],axis = 1)-df.drop(['quality','PoorQuality'],axis = 1).mean())/df.drop(['quality','PoorQuality'],axis = 1).std()

normalized_df
X = normalized_df[:]

y = df['PoorQuality']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

pred_logreg = logreg.predict(X_test)

print(accuracy_score(pred_logreg,y_test))

print(classification_report(pred_logreg,y_test))

print(confusion_matrix(pred_logreg,y_test))
rtree = RandomForestClassifier(n_estimators=100)

rtree.fit(X_train,y_train)

pred_rtree = logreg.predict(X_test)

print(accuracy_score(pred_rtree,y_test))

print(classification_report(pred_rtree,y_test))

print(confusion_matrix(pred_rtree,y_test))
error_rate = []

for i in range(1,40):

    count = 0

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    for i in range(0,len(pred_i)):

        if np.array(pred_i)[i]!=np.array(y_test)[1]:

            count+=1

    error_rate.append(count)
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Incorrect Prediction vs. K Value')

plt.xlabel('K')

plt.ylabel('Incorrect Prediction')
knn = KNeighborsClassifier(n_neighbors=37)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=37')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))

print(accuracy_score(y_test,pred))