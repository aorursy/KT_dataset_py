import numpy as np

import pandas as pd



dataset = pd.read_csv('../input/Iris.csv')

x = dataset.iloc[:, 1:5].values

y = dataset.iloc[:, 5].values



from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
#Testing the model predictions

print(cm)

print(y_pred)

print(y_test)