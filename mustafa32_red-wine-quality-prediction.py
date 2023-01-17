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



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,LabelEncoder



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier

from sklearn.svm import SVC,LinearSVC



from sklearn.metrics import accuracy_score,r2_score,classification_report
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), cmap="YlGnBu")
# plt.figure(figsize=(15,18))

sns.barplot(df['quality'],df['citric acid'],palette="Blues_d")
sns.barplot(df['quality'],df['fixed acidity'])
plt.figure(figsize=(15,10))

df['quality'].value_counts().plot.pie()
sns.countplot(df['quality'])
sns.barplot(df['quality'],df['residual sugar'])
sns.barplot(df['quality'],df['sulphates'])
sns.barplot(df['quality'],df['alcohol'])
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

label_quality = LabelEncoder()

df['quality'] = label_quality.fit_transform(df['quality'])
X = df.drop('quality',axis=1)

y = df['quality']

X = StandardScaler().fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = []
linearreg = LogisticRegression()

linearreg.fit(X_train,y_train)

y_predict = linearreg.predict(X_test)

model.append(["Logistic",accuracy_score(y_predict,y_test)])
linearreg = RandomForestClassifier(n_estimators=700,n_jobs=-1)

linearreg.fit(X_train,y_train)

y_predict = linearreg.predict(X_test)

model.append(["RandomForestClassifier",accuracy_score(y_predict,y_test)])
linearreg = GaussianNB()

linearreg.fit(X_train,y_train)

y_predict = linearreg.predict(X_test)

model.append(["GaussianNB",accuracy_score(y_predict,y_test)])
linearreg = SVC()

linearreg.fit(X_train,y_train)

y_predict = linearreg.predict(X_test)

model.append(["SVC",accuracy_score(y_predict,y_test)])
linearreg = ExtraTreesClassifier()

linearreg.fit(X_train,y_train)

y_predict = linearreg.predict(X_test)



model.append(["ExtraTreeClassifier",accuracy_score(y_predict,y_test)])
linearreg = DecisionTreeClassifier()

linearreg.fit(X_train,y_train)

y_predict = linearreg.predict(X_test)

model.append(["DecisionTreeClassifier",accuracy_score(y_predict,y_test)])
model = pd.DataFrame(model)

model.columns = ["Model Name","Score"]

model.sort_values(by="Score",ascending=False)