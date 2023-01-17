from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
table_max = pd.read_csv("../input/turkey-climate-data/average_max_temp.csv").sort_values(by=["City"]).reset_index(drop=True)
table_min = pd.read_csv("../input/turkey-climate-data/average_min_temp.csv").sort_values(by=["City"]).reset_index(drop=True)
table_prep = pd.read_csv("../input/turkey-climate-data/average_precipitation.csv").sort_values(by=["City"]).reset_index(drop=True)

turkey_map = gpd.read_file("../input/covid19-in-turkey-by-regions/shape/turkey.shp").sort_values(by=["City"])
features = []
for column in table_max.columns[1:-1]:
    features.append(table_max[column])
    
for column in table_min.columns[1:-1]:
    features.append(table_min[column])
    
for column in table_prep.columns[1:-1]:
    features.append(table_prep[column] / 10)
    
train_X = np.column_stack(features)
model = KMeans(n_clusters=7, init='random', n_init=10, max_iter=100, tol=1e-04, random_state=0).fit(train_X)
turkey_map["Label"] = model.labels_
turkey_map.plot(column="Label", figsize=(20, 8))

plt.show()
