import pandas as pd

import matplotlib.pyplot as plt
dataset=pd.read_csv("../input/cereal.csv")

print(dataset)

dataset['fat'].plot.hist()

plt.title("Histogram of fat")