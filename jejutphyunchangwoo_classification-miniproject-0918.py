import numpy as np

import pandas as pd
dataset = pd.read_csv('../input/iris/Iris.csv')
dataset.head()
dataset['Species'].value_counts()
dataset.describe()
X = dataset.iloc[:,1:5].values

y = dataset.iloc[:,-1].values
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

y = lb.fit_transform(y)
y
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
cm
acc = accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
acc