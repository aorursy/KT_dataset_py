import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
df.drop('objid',axis = 1, inplace = True)
df.head()
plt.figure(figsize = (10,6))
sns.heatmap(df.isnull()==False)
df.info()
plt.figure(figsize = (10,6))
sns.boxplot(y = 'ra', x = 'class', data = df)
plt.figure(figsize = (10,6))
sns.countplot(x = 'class', data = df)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'u', data = df)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'g', data = df)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'r', data = df)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'i', data = df)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'z', data = df)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'redshift', data = df)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'class', y = 'mjd', data = df)
from sklearn.cross_validation import train_test_split
y = df['class']
x = df.drop('class', axis = 1)
x = x.drop('ra', axis = 1)
x = x.drop('dec', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))