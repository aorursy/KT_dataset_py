%matplotlib inline
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
dataset = sns.load_dataset("iris")
dataset.head(100)
dataset.columns
sns.pairplot(dataset,hue="species")
data_train=dataset.sample(frac=0.8,random_state=1)
data_test=dataset.drop(data_train.index)
X_train = data_train.drop(['species'], axis=1)
y_train = data_train['species']
X_test = data_test.drop(['species'], axis=1)
y_test = data_test['species']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
print(accuracy_score(y_test, y_lr))
print(confusion_matrix(y_test, y_lr))
pd.crosstab(y_test, y_lr, rownames=['Reel'], colnames=['Prediction'], margins=True)
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_dtc = dtc.predict(X_test)
print(accuracy_score(y_test, y_dtc))
print(confusion_matrix(y_test, y_dtc))
pd.crosstab(y_test, y_dtc, rownames=['Reel'], colnames=['Prediction'], margins=True)
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
print(accuracy_score(y_test, y_rf))
print(confusion_matrix(y_test, y_rf))
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(6,6))
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance des caracteristiques')