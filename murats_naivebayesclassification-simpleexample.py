import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/breastCancer.csv")
df.head()
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
df.head()
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
df.diagnosis.tail()
y = df.diagnosis.values

x_data = df.drop(["diagnosis"], axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state=1)
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x_train, y_train)
print("accuracy of naive bayes algorithm ", model.score(x_test,y_test))