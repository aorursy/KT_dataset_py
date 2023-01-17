import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

df.info()
df.head()
df.isnull().sum()
x = df.iloc[:,0:8].values

y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler

standardscaler = StandardScaler()
x_train = standardscaler.fit_transform(x_train)

x_test = standardscaler.transform(x_test)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))