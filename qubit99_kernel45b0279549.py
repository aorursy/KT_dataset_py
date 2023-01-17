import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
data=pd.read_csv('../input/cardio_train.csv',sep=';',header=0)
data.head()
data=data.dropna()
print(data.shape)
print(data.columns)
len(data.columns)
data.drop(columns=['id'])
data.cardio.value_counts()

sns.countplot(x = 'cardio', data = data, palette = 'hls')

plt.show()
X=data[[

 'age',

 'gender',

 'height',

 'weight',

 'ap_hi',

 'ap_lo',

 'cholesterol',

 'gluc',

 'smoke',

 'alco',

 'active']]

y=data['cardio']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)
y_pred=(logreg.predict_proba(X_test)[:,1] >= 0.53).astype(bool)

y_pred
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))