import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
data = pd.read_csv('../input/mushrooms.csv')

data.describe()
data.info()
data['cap-shape'].unique()
le = preprocessing.LabelEncoder()



for column in data.columns:

    data[column] = le.fit_transform(data[column])
data.info()
print(data.groupby('class').size())
y = data['class'].values

data.drop('class',axis=1,inplace=True)

y
reg = preprocessing.StandardScaler()

data = reg.fit_transform(data)
X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.2,random_state=0)
nn = MLPClassifier()

nn.fit(X_train,y_train)
y_pred = nn.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))