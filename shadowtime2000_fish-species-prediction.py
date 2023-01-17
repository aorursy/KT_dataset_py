import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
x = pd.read_csv("/kaggle/input/fish-market/Fish.csv", usecols=["Height", "Length1", "Length2", "Length3", "Width"]).to_numpy()
y = pd.read_csv("/kaggle/input/fish-market/Fish.csv", usecols=["Species"]).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)

sc = StandardScaler()
sc.fit(x_train)

x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

encoder = OneHotEncoder()
encoder.fit(y_train)

y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()
model = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100), solver="lbfgs", max_iter=1000) # This is may take too much processing power
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test, y_test))
import pickle

with open("model.pkl", "wb+") as f:
    pickle.dump(model, f)
print("Model successfully saved!")
vertical_length = float(input("Vertical Length: "))
diagonal_length = float(input("Diagonal Length: "))
cross_length = float(input("Cross Length: "))
height = float(input("Height: "))
width = float(input("Width: "))

scaled = sc.transform([[vertical_length, diagonal_length, cross_length, height, width]])
prediction_raw = model.predict(scaled)

print(encoder.inverse_transform(prediction_raw)) ## Returns 2D Array you will have to convert more from their.