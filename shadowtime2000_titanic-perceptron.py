import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
training_data = pd.read_csv("/kaggle/input/titanic/train.csv", usecols=["Pclass", "SibSp", "Parch"])

training_labels = pd.read_csv("/kaggle/input/titanic/train.csv", usecols=["Survived"])

testing_data = pd.read_csv("/kaggle/input/titanic/test.csv", usecols=["Pclass", "SibSp", "Parch"])

passenger_id = pd.read_csv("/kaggle/input/titanic/test.csv", usecols=["PassengerId"])
x_train = training_data.to_numpy()

y_train = training_labels.to_numpy()

test = testing_data.to_numpy()



sc = StandardScaler()

sc.fit(x_train)



x_train = sc.transform(x_train)

test = sc.transform(test)
model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000, activation="relu", solver="lbfgs", random_state=1)

model.fit(x_train, y_train)
predictions = model.predict(test)



output = pd.DataFrame({"PassengerId": passenger_id.PassengerId, "Survived": predictions})

output.to_csv("submission.csv", index=False)

print("Submission Saved!")