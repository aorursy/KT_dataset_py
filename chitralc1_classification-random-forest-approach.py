import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_palette("rainbow")

df = pd.read_csv('../input/Iris.csv')
df.head()
df.info()
df.columns
sns.pairplot(df,hue='Species')
sns.countplot(x = 'Species', data = df)
sns.boxplot(x= 'Species',y = 'PetalLengthCm', data = df)
sns.boxplot(x= 'Species',y = 'PetalWidthCm', data = df)
sns.heatmap(df.corr(), annot=True)
from sklearn.model_selection import train_test_split
X = df.drop('Species',axis=1)

y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,rfc_pred))