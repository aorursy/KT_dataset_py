import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

data = pd.read_csv('../input/pid-5M.csv')
data.head()
model = KMeans(n_clusters=4)
data = data.drop(['id'], axis=1)
data.head()
model.fit(data)
prediction1 = model.predict([[0.78, 1.08, 0.99, 0, 0.0, 0.0]])
print(prediction1)