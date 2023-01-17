import pandas as pd
pd.set_option("max_columns", None)
df = pd.read_csv("../input/recipeData.csv", encoding='latin-1')

from sklearn.preprocessing import StandardScaler
import yellowbrick as yb

df = df.dropna(subset=['Size(L)', 'OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime'])
X = df.iloc[:, 6:6+7]
trans = StandardScaler()
trans.fit(X)
X = pd.DataFrame(trans.transform(X), columns=X.columns)
y = (df['Style'] == 'American IPA').astype(int)

df.head()
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

import numpy as np
np.random.seed(42)
X_sample = X.sample(5000)
y_sample = y.iloc[X_sample.index.values]

vzr = KElbowVisualizer(KMeans(), k=(4, 10))
vzr.fit(X_sample)
vzr.poof()
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=8)
from yellowbrick.cluster import SilhouetteVisualizer

clf = KMeans(n_clusters=8)
vzr = SilhouetteVisualizer(clf)
vzr.fit(X)
vzr.poof()