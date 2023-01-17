import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report
data = pd.read_csv("../input/mushrooms.csv")
data.head()
data.describe()
data = data.apply(LabelEncoder().fit_transform)
X = data.iloc[:,1:]
y = data.iloc[:, 0]
for col in X.columns:

    X[col].astype('category')
X.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
svc = SVC()
svc.fit(X_train, y_train)
p = svc.predict(X_test)
print(classification_report(y_test, p))