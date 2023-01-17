import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/churn-modellingcsv/Churn_Modelling.csv')
df.head()
X= df.iloc[:,3:13].values

Y = df.iloc[:,13].values
X
Y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X[:,1] = le.fit_transform(X[:,1])# Burada ulkeleri surekli degiskene ceviriyoruz.



le2 = LabelEncoder()

X[:,2] = le2.fit_transform(X[:,2])# Burada cinsiyetleri surekli degiskene ceviriyoruz.
X
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories= 'auto')

X=ohe.fit_transform(X).toarray()

X = X[:,1:]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
from xgboost import XGBClassifier
classifier = XGBClassifier()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))