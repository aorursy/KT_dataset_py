import numpy as np 

import pandas as pd
data = pd.read_csv('../input/diabetes.csv')
data.head()
data.info()
data.describe()
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')
sns.countplot(x='Outcome', data=data, palette='Set1')
plt.figure(figsize=(8,6))

sns.heatmap(data.corr(), cmap='GnBu')
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()
for colname in data.columns[:-1]:

    newcolname = 'norm'+colname

    data[colname] = data[colname].astype('float64')

    data[newcolname] = ss.fit_transform(data[colname].values.reshape(-1, 1))
data.head()
data.drop(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], axis=1, inplace=True)



data = data.rename(columns={'Outcome': 'class'})



data.head()
plt.figure(figsize=(8,6))

sns.heatmap(data.corr(), cmap='GnBu')
sns.pairplot(data)
from sklearn.model_selection import train_test_split
X = data.drop(['class'], axis=1)

y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
from tpot import TPOTClassifier
clf = TPOTClassifier(generations=10, population_size=50, n_jobs=-1, random_state=101, verbosity=2)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))