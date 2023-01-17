import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
dataset.tail()
y=dataset.iloc[0:1599,11].values

y=np.where(y>=7,1,0)
x = dataset.iloc[0:1599, [1, 10]].values
print('Class labels:', np.unique(y))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100,criterion = 'entropy')

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)
accuracy