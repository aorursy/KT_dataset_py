import numpy as np

import pandas as ps

import matplotlib.pyplot as plt
# Reading the input

data = ps.read_csv('../input/train.csv')
# print the first 10 rows

data.info()
# printing the first ten values

data.head(10)
age = data.Age

p_class = data.Pclass

sex = data.Sex

surv = data.Survived
plt.rcdefaults()
counts, bins = np.histogram(age, bins=10, range=(0, 100))

print(counts)

print(bins)
plt.hist(age, bins=10,range = (0,100))



plt.title("Histogram Showing Age Distribution of Passangers in Titanic")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()