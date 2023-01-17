import pandas as pd

import matplotlib.pyplot as plt

nutrition = pd.read_csv("../input/cereal.csv")

nutrition.describe()



plt.hist(nutrition["protein"])

plt.title("Protein in Cereals")