import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
df = pd.read_csv("../input/heart.csv")

df.head()
df.info()
df.describe()
df.shape
df.target.value_counts()
#checking correlation of each feature



plt.figure(figsize=(12,7))

sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.figure(figsize=(10,7))

sns.set_style('whitegrid')

sns.countplot('target', data=df)
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])
dataset.head()
X = dataset.drop('target', axis = 1)



y = df['target']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.model_selection import cross_val_score
cross_val_score(RandomForestClassifier(n_estimators = 60), X, y)
cross_val_score(DecisionTreeClassifier(), X,y)
cross_val_score(KNeighborsClassifier(n_neighbors=15), X,y)
cross_val_score(GaussianNB(), X,y)
cross_val_score(LogisticRegression(), X,y)
# It seems like logistic regression algorithm performs well than another with good accuracy.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

pred
from sklearn.metrics import confusion_matrix
confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames =['prediction'])

confusion_matrix
plt.figure(figsize=(12,7))



plt.title('Confusion Matrix')



sns.heatmap(confusion_matrix, annot = True, cmap='coolwarm')
from sklearn.metrics import classification_report

report = classification_report(y_test, pred)

print(report)
lr.score(X_test, y_test)