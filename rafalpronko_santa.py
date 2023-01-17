import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('../input/gifts.csv')
df.head()
clf = KMeans(n_clusters = 1410, n_jobs=-1, verbose=1)
pred = clf.fit_transform(df[['Latitude', 'Longitude', 'Weight']])
