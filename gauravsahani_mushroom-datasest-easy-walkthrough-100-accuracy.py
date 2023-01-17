import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

Encoder=LabelEncoder()
for col in df.columns:

    df[col]=Encoder.fit_transform(df[col])
df.head()
df.dtypes
plt.hist(df['class'])
X=df.iloc[:,1:23]

y=df[['class']]
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train[0]
from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier()

RF.fit(X_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix

pred=RF.predict(X_test)



print(accuracy_score(pred,y_test))

print(confusion_matrix(pred,y_test))
pred