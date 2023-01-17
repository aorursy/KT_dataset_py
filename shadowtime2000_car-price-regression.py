import numpy as np

import pandas as pd



import pickle



from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
data = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv", usecols=["year", "mileage", "mpg","engineSize"]).to_numpy() # You can probably change the car to whatever you want

labels = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv", usecols=["price"]).to_numpy()



x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=30)



sc1 = StandardScaler()

sc1.fit(x_train)



x_train = sc1.transform(x_train)

x_test = sc1.transform(x_test)



sc2 = StandardScaler()

sc2.fit(y_train)



y_train = sc2.transform(y_train)

y_test = sc2.transform(y_test)
model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=500, solver="lbfgs")

model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test, y_test))
with open("model.pkl", "wb") as f:

    pickle.dump(model, f)

print("Model successfully saved!")
year = float(input("Registration year: "))

mileage = float(input("Mileage: "))

mpg = float(input("MPG: "))

engine_size = float(input("Engine Size: "))



data = np.array([year, mileage, mpg, engine_size]).reshape(1, 4)



data = sc1.transform(data)



prediction = model.predict(data)



prediction = sc2.inverse_transform(prediction)



print("Predicted price for car: ", prediction[0])