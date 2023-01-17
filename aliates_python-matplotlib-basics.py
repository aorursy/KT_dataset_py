import matplotlib.pyplot as plt

import pandas as pd
dataFrame = pd.read_csv("../input/iris.csv")



print("dataFrame.info():")

print(dataFrame.info(), "\n")

print("dataFrame.describe():\n{0}\n".format(dataFrame.describe()))
print("dataFrame.head(10):\n{0}\n".format(dataFrame.head(10)))
print("dataFrame.Species.unique(): [Unique \"Species\" classes]\n{0}\n"

      .format(dataFrame.Species.unique()))
setosa_df = dataFrame[dataFrame.Species == "Iris-setosa"]

print("setosa_df:\n{0}\n".format(setosa_df.head(10)))

print("setosa_df.describe():\n{0}\n".format(setosa_df.describe()))
versicolor_df = dataFrame[dataFrame.Species == "Iris-versicolor"]

print("versicolor_df:\n{0}\n".format(versicolor_df.head(10)))

print("versicolor_df.describe():\n{0}\n".format(versicolor_df.describe()))
virginica_df = dataFrame[dataFrame.Species == "Iris-virginica"]

print("virginica_df:\n{0}\n".format(virginica_df.head(10)))

print("virginica_df.describe():\n{0}\n".format(virginica_df.describe()))
#Default plot



plt.plot(setosa_df.Id, setosa_df.PetalLengthCm, color="red", label="Setosa")

plt.plot(versicolor_df.Id, versicolor_df.PetalLengthCm, color="green", label="Versicolor")

plt.plot(virginica_df.Id, virginica_df.PetalLengthCm, color="blue", label="Virginica")



plt.legend()

plt.xlabel("ID")

plt.ylabel("PETAL-LENGTH-CM")

plt.show()
#Scatter plot



plt.scatter(setosa_df.Id, setosa_df.PetalLengthCm, color="red", label="Setosa")

plt.scatter(versicolor_df.Id, versicolor_df.PetalLengthCm, color="green", label="Versicolor")

plt.scatter(virginica_df.Id, virginica_df.PetalLengthCm, color="blue", label="Virginica")



plt.legend()

plt.xlabel("ID")

plt.ylabel("PETAL-LENGTH-CM")

plt.show()
#Histogram plot



plt.hist(setosa_df.PetalLengthCm, bins=50)

plt.xlabel("PetalLengthCm")

plt.ylabel("Frequency")

plt.title("Histogram")

plt.show()
#Bar plot

import numpy as np



x = np.array([1, 2, 3, 4, 5])

y = np.array(["A", "B", "C", "D", "E"])



plt.bar(x, y)

plt.title("Bar Plot")

plt.xlabel("X")

plt.ylabel("Y")

plt.show()
#Sub plot



setosa_df.plot(grid=True, alpha=0.9, subplots=True)

plt.show()
#Sub plot



plt.subplot(2, 1, 1)

plt.plot(setosa_df.Id, setosa_df.PetalLengthCm, color="red", label="Setosa")

plt.ylabel("Setosa-PetalLengthCm")



plt.subplot(2, 1, 2)

plt.plot(versicolor_df.Id, versicolor_df.PetalLengthCm, color="green", label="Versicolor")

plt.ylabel("Versicolor-PetalLengthCm")



plt.show()