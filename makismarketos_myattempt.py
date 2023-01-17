import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
df_train = pd.read_csv("../input/train.csv")
centroids = df_train[["x","y","place_id"]].groupby("place_id").median()

cent_small = centroids[(centroids["x"] < 0.125) & (centroids["y"] < 0.125)]
cent_small
vor = Voronoi(cent_small, qhull_options="Qc")

voronoi_plot_2d(vor)
plt.show()
