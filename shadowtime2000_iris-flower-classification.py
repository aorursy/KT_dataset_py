import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
x = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv", usecols=["sepal_length", "sepal_width", "petal_length", "petal_width"]).to_numpy()

y = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv", usecols=["species"]).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30)



sc = StandardScaler()

sc.fit(x_train)



x_train = sc.transform(x_train)

x_test = sc.transform(x_test)



encoder = OneHotEncoder()

encoder.fit(y_train)



y_train = encoder.transform(y_train)

y_test = encoder.transform(y_test)
model = MLPClassifier(hidden_layer_sizes=(100, 100, 50), solver="lbfgs", max_iter=500)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))