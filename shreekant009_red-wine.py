import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
df.describe()
df.columns
df.isnull().sum()
sns.barplot(x='quality',y='citric acid',data=df)
sns.barplot(x='quality',y='volatile acidity',data=df)
sns.barplot(x='quality',y='sulphates',data=df)
sns.barplot(x='quality',y='alcohol',data=df)
sns.barplot(x='quality',y='free sulfur dioxide',data=df)
sns.barplot(x='quality',y='fixed acidity',data=df)
for i in range(len(df)):

    if df.iloc[i,11]>=6.5:

        df.iloc[i,11]='good'

    else:

        df.iloc[i,11]='bad'
sns.barplot(x='quality',y='alcohol',data=df)
df.head()
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()
df['quality'] = labelEncoder_X.fit_transform(df['quality'])
df.head()
y=df['quality']

X=df.drop('quality',axis=1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
#Logistic Regression

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
print(accuracy_score(y_test,predictions ))

print(confusion_matrix(y_test,predictions ))
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc__pred = rfc.predict(X_test)
print(accuracy_score(y_test,rfc__pred))

print(confusion_matrix(y_test,rfc__pred))