import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df.head()
sample = df.iloc[4].drop("label")

sample = sample.values.reshape(28,28)
plt.matshow(sample, cmap="gray")

plt.show()
y = df["label"].values

x = df.drop("label", axis=1)
print("Shape of X: ", x.shape)

print("Shape of Y: ", y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier()

rf_model.fit(x_train, y_train)
predictions = rf_model.predict(x_test)
print(f1_score(y_test, predictions, average="weighted"))

print("Accuracy:", accuracy_score(y_test, predictions))
y_test[15]
plt.matshow(x_test.iloc[1222].values.reshape(28, 28), cmap="gray")
rf_model.predict(x_test.iloc[1222].values.reshape(1, -1))