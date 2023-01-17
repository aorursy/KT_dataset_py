import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("../input/wine-dataset/wine.csv")
data_scaled = StandardScaler().fit_transform(data)
pca = PCA(n_components=3)
pca.fit(data_scaled)
data_transformed = pca.transform(data_scaled)