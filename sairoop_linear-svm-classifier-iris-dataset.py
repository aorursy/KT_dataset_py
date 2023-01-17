import numpy as np

import pandas as pd



dataset = pd.read_csv("../input/Iris.csv")

x = dataset.iloc[:, 1:4].values

y = dataset.iloc[:, -1].values



from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()

x = scaler_x.fit_transform(x)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
from sklearn.svm import SVC

classifier = SVC(kernel='linear')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)