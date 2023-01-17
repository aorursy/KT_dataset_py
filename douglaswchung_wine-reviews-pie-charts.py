import pandas as pd
reviews = pd.read_csv("../input/winemag-data_first150k.csv", index_col=0)
reviews['province'].value_counts().head(3).plot.pie()

# Unsquish the pie.
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal')