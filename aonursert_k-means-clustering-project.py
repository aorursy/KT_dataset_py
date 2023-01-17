import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/us-news-and-world-reports-college-data/College.csv", index_col=0)
df.head()
df.info()
df.describe()
fig = plt.figure(figsize=(10,6))
sns.scatterplot(x="Room.Board", y="Grad.Rate", data=df, hue="Private")
fig = plt.figure(figsize=(10,6))
sns.scatterplot(x="Outstate", y="F.Undergrad", data=df, hue="Private")
sns.set_style("whitegrid")
g = sns.FacetGrid(df, hue="Private", height=5)
g.map(plt.hist, "Outstate")
g = sns.FacetGrid(df, hue="Private", height=5)
g.map(plt.hist, "Grad.Rate")
df[df["Grad.Rate"] > 100]
df.loc["Cazenovia College", "Grad.Rate"] = 100
df[df["Grad.Rate"] > 100]
g = sns.FacetGrid(df, hue="Private", height=5)
g.map(plt.hist, "Grad.Rate")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop("Private", axis=1))
kmeans.cluster_centers_
df = pd.get_dummies(df, columns=["Private"], drop_first=True)
df.head()
df = df.rename(columns={'Private_Yes': 'Cluster'})
df.head()
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df["Cluster"], kmeans.labels_))
print(classification_report(df["Cluster"], kmeans.labels_))